# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:00:00 2022

@author: hdb

"""

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torchvision
import numpy as np
import math


class DCN_(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, 
                 padding=0, dilation=1, deformable_groups=1):
        super(DCN, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True) # 原卷积
        
        channel_offset = 2 * kernel_size[0] * kernel_size[1] * deformable_groups
        self.conv_offset = nn.Conv2d(in_channels, channel_offset, kernel_size=kernel_size, stride=stride, padding=padding)  # out_channel = 2 * offset_groups * kernel_height * kernel_width
        init_offset = torch.Tensor(np.zeros([channel_offset, in_channels, kernel_size[0], kernel_size[1]]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset) #初始化为0
 
        channel_mask = kernel_size[0] * kernel_size[1] * deformable_groups
        self.conv_mask = nn.Conv2d(in_channels, channel_mask, kernel_size=kernel_size, stride=stride, padding=padding)
        init_mask = torch.Tensor(np.zeros([channel_mask, in_channels, kernel_size[0], kernel_size[1]]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask) #初始化为0.5
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))   #保证在0到1之间
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                            bias=self.conv.bias,
                                            stride=[self.stride, self.stride],
                                            padding=[self.padding, self.padding],
                                            dilation=[self.dilation, self.dilation],
                                            mask=mask)
        return out



class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
               offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
               mask.shape[1]
        return torchvision.ops.deform_conv2d(input=input, 
                                             offset=offset,
                                             weight=self.weight,
                                             bias=self.bias,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation,
                                             mask=mask)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(input=input, 
                                             offset=offset,
                                             weight=self.weight,
                                             bias=self.bias,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation,
                                             mask=mask)
