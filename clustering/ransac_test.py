import cv2 as cv
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath("./"))
from load_h5py_files import load_sift_output
sys.path.append(os.path.abspath("./feature_matching"))
from flann_matcher import convert_result_into_opencv
                

def get_disk_output(output, image_name):
  return next(filter(lambda x: x["image_name"] == image_name, output), None)

def run_ransac(matcher_result, desc_filepath, desc_type, min_point=8, reproj_thres=1.5, confidence=0.99):
    ransac_result = []

    for result in matcher_result:
        pair = result["pair"]
        good_matches = result["good_matches"]
        image0 = result["images_name"][0]
        image1 = result["images_name"][1]
        
        if len(good_matches) < min_point: # based on opencv documentation, to use RANSAC it needs min. 8 points
            continue

        if desc_type == "sift":
            kps0 = load_sift_output(desc_filepath, image0)[1]
            kps1 = load_sift_output(desc_filepath, image1)[1]
        elif desc_type == "disk":
            disk_output = torch.load(desc_filepath, weights_only=False)
            features0 = get_disk_output(disk_output, image0)
            features1 = get_disk_output(disk_output, image1)
        
            kps0 = convert_result_into_opencv(features0)["keypoints"]
            kps1 = convert_result_into_opencv(features1)["keypoints"]

        pts0 = np.float32([kps0[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        pts1 = np.float32([kps1[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_RANSAC, reproj_thres, confidence)
    
        if F is None or mask is None:
            continue
    
        inlier = [m for i,m in enumerate(good_matches) if mask[i][0] == 1]
        ransac_result.append({
            "pair": pair,
            "images_name" : [image0, image1],
            "good_matches" : inlier,
        })

    return ransac_result