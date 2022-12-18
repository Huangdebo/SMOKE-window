# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:02:59 2022

@author: hdb
"""

import numpy as np
import cv2
from PIL import ImageDraw, Image
import os
import csv
import argparse
import platform
import torch

# from smoke.utils import comm
# from smoke.utils.miscellaneous import mkdir
# from smoke.utils.logger import setup_logger
# from smoke.utils.collect_env import collect_env_info
# from smoke.utils.envs import seed_all_rng

from smoke.modeling.heatmap_coder import get_transfrom_matrix
from smoke.utils.check_point import DetectronCheckpointer
from smoke.modeling.detector import build_detection_model
from smoke.modeling.smoke_coder import SMOKECoder
from smoke.structures.params_3d import ParamsList
from smoke.data.transforms import build_transforms
from smoke.utils.plot import plot_box
from smoke.config import cfg

from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)

def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}
def load_annotations(cfg, args, file_name, is_train=False):
    annotations = []
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                  'dl', 'lx', 'ly', 'lz', 'ry']

    classes = cfg.DATASETS.DETECT_CLASSES
    label_dir = os.path.join(args.data_root, "label_2")
    calib_dir = os.path.join(args.data_root, "calib")
    if is_train:
        with open(os.path.join(label_dir, file_name) + '.txt', 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

            for line, row in enumerate(reader):
                if row["type"] in classes:
                    annotations.append({
                        "class": row["type"],
                        "label": TYPE_ID_CONVERSION[row["type"]],
                        "truncation": float(row["truncated"]),
                        "occlusion": float(row["occluded"]),
                        "alpha": float(row["alpha"]),
                        "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                        "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                        "rot_y": float(row["ry"])
                    })

    # get camera intrinsic matrix K
    with open(os.path.join(calib_dir, file_name) + '.txt', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                K = row[1:]
                K = [float(i) for i in K]
                K = np.array(K, dtype=np.float32).reshape(3, 4)
                K = K[:3, :3]
                break

    return annotations, K

def detect_img(model, cfg, args, device):
    file_name = '000285'
 
    input_width = 1280
    input_height = 384
    down_retio = 4
    output_width = input_width // down_retio
    output_height = input_height // down_retio

    img_path = os.path.join(args.data_root, 'image_2', file_name) +  '.png'
    img_org = Image.open(img_path)
    center = np.array([i / 2 for i in img_org.size], dtype=np.float32)
    size = np.array([i for i in img_org.size], dtype=np.float32)

    center_size = [center, size]
    trans_affine = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)
    img = img_org.transform(
        (input_width, input_height),
        method=Image.Transform.AFFINE,
        data=trans_affine_inv.flatten()[:6],
        resample=Image.Resampling.BILINEAR,
    )
    trans_mat = get_transfrom_matrix(
        center_size,
        [output_width, output_height]
    )
    
    annotations, K = load_annotations(cfg, args, file_name)
    
    target = ParamsList(image_size=size,
                        is_train=False)
    target.add_field("trans_mat", trans_mat)
    target.add_field("K", K)
    
    transforms = build_transforms(cfg, is_train=False)
    img, target = transforms(img, target)
    img = img.to(device)
    target = [target]

    model.eval()
    output = model(img, target)
    
    smoke_coder = SMOKECoder(
        cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
        cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        cfg.MODEL.DEVICE,
    )
    draw = ImageDraw.Draw(img_org)

    clses = output[:, 0:1]
    pred_alphas = output[:, 1:2]
    box2d = output[:, 2:6]
    pred_dimensions = output[:, 6:9]  # h,w,l
    pred_locations = output[:, 9:12]
    pred_rotys = output[:, 12:13]
    scores = output[:, 13:14]
    
    pred_dimensions_lhw = pred_dimensions.roll(shifts=1, dims=1)  # change dimension back to l,h,w
    box3d = smoke_coder.encode_box3d_img(K, pred_rotys, pred_dimensions_lhw, pred_locations)
    
    plot_box(draw, None, box3d)
        
    img_org.show()




def detect_argument_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default="./configs/smoke_gn_vector.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default='./weights/kitti/DLA/model_final.pth',
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--data-root", type=str, default='./datasets/kitti/testing', help="the root path of the data")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    plat = platform.system().lower()
    if plat == 'windows':
        port = 100
    else:
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    # torch.cuda.set_device(0)
    args = detect_argument_parser().parse_args()
    cfg_det = setup(args)
    
    model = build_detection_model(cfg_det)
    device = torch.device(cfg_det.MODEL.DEVICE)
    model.to(device)
    
    checkpointer = DetectronCheckpointer(
        cfg_det, model, save_dir=cfg_det.OUTPUT_DIR
    )
    _ = checkpointer.load(args.ckpt, use_latest=args.ckpt is None)
    detect_img(model, cfg_det, args, device)
    