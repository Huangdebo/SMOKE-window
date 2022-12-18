
#  window 版 SMOKE

![image](figures/result.gif)

##  添加功能

- 省去 linux 下编译 DConv 的 cuda 代码，可以直接在 window 下训练和测试
- 改变训练循环，并使用梯度累加机制
- 增添了 finetune 和 resume 等功能
- 提供了测试单张图像的例子


## 训练

1. 下载 KITTI 数据，并修改成一下结构，把图像列表放在 ImageSets 文件夹中，然后在 datasets 中创建 kitti 目录的软连接
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```
     
2. 执行 tools/plain_train_net.py 即可

3. 项目3
