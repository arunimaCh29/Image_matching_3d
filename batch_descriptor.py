from typing import Literal
import torch
from torch.utils.data import DataLoader
from feature_matching.LightGlue.lightglue import LightGlue, DISK, SIFT
from feature_matching.LightGlue.lightglue.utils import load_image, rbd
# from feature_matching.lightglue_matcher import get_SIFT_features, get_DISK_features
# import cv2 as cv
import h5py
import os
import numpy as np

def get_SIFT_features(image, device, cuda, max_keypoints):
    if device:
        extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)  # load the extractor
        # image = load_image(image_location).cuda()
    else:
        extractor = SIFT(max_num_keypoints=max_keypoints).eval()  # load the extractor
        # image = load_image(image_location)

    feats = extractor({"image": image})  # auto-resize the image, disable with resize=None

    return feats

def save_result(save_dir, extractor, i, dataset_name, scene_name, image_name, image_path, res):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{i}_1024_{extractor}.h5")
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('keypoints', data=res['keypoints'].cpu().detach().numpy())
        f.create_dataset('keypoint_scores', data=res['keypoint_scores'].cpu().detach().numpy())
        f.create_dataset('descriptors', data=res['descriptors'].cpu().detach().numpy())
        f.create_dataset('image_name', data=image_name, dtype=dt)
        f.create_dataset('dataset_name', data=dataset_name, dtype=dt)
        f.create_dataset('scene_name', data=scene_name, dtype=dt)
        f.create_dataset('image_path', data=image_path, dtype=dt)

def batch_feature_descriptor(loader, device, descriptor_type, output_dir, max_keypoints=2048, cuda=True):
    for i, batch in enumerate(loader):
        if descriptor_type == "sift":
            features = get_SIFT_features(batch["image"].to(device), device, cuda, max_keypoints)
            save_result(output_dir, 'sift', i, batch['dataset_name'], batch['scene_name'], batch['image_name'], batch['image_path'], features)

            del features