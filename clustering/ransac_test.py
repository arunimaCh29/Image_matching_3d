import cv2 as cv
import numpy as np
import torch
import os
import sys

                

def get_disk_output(output, image_name):
  return next(filter(lambda x: x["image_name"] == image_name, output), None)

def run_ransac(points0, points1, min_point=8, reproj_thres=1.5, confidence=0.99):

    pts0 = points0.cpu().numpy()
    pts1 = points1.cpu().numpy()

        
    if pts0.shape[0] < min_point: # based on opencv documentation, to use RANSAC it needs min. 8 points
        return None,None,None
    
    # Find fundamental matrix with RANSAC
    F, mask = cv.findFundamentalMat(
        pts0, pts1, 
        method=cv.FM_RANSAC,
        ransacReprojThreshold=reproj_thres,
        confidence=confidence
    )
    
    if F is None or mask is None:
        mask = np.zeros(len(pts0), dtype=bool)
        return None, None, None
    
    # Convert mask to boolean array
    mask = mask.ravel().astype(bool)
    
    # Filter points and matches using the mask
    filtered_points0 = pts0[mask]
    filtered_points1 = pts1[mask]
    #filtered_matches = matches[mask]
    
    return filtered_points0, filtered_points1, mask