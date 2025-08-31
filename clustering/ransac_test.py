import cv2 as cv
import numpy as np
import torch
import os
import sys
# sys.path.append(os.path.abspath("./"))
# from load_h5py_files import load_sift_output
# sys.path.append(os.path.abspath("./feature_matching"))
# from flann_matcher import convert_result_into_opencv
                

# def get_disk_output(output, image_name):
#   return next(filter(lambda x: x["image_name"] == image_name, output), None)

# copy from branch evaluation_feature
def run_ransac(points0, points1, min_point=8, reproj_thres=1.5, confidence=0.99):
    '''
    Run ransac to remove noisy keypoints
    Args:
        points0 (torch.Tensor): matched keypoints from first image
        points1 (torch.Tensor): matched keypoints from second image
        min_point (int): minimum length of matched keypoints
        reproj_thres (float): parameter for openCV ransac
        confidence (float): parameter for openCV ransac

    Returns:
        numpy.ndarray: filtered matched points0 after ransac, None if matched keypoints smaller than minimum point or no result from ransac
        numpy.ndarray: filtered matched points1 after ransac, None if matched keypoints smaller than minimum point or no result from ransac
        numpy.ndarray: ransac mask, None if matched keypoints smaller than minimum point or no result from ransac
    '''

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

# def run_ransac(matcher_result, desc_filepath, desc_type, min_point=8, reproj_thres=1.5, confidence=0.99):
#     ransac_result = []

#     for result in matcher_result:
#         pair = result["pair"]
#         good_matches = result["good_matches"]
#         image0 = result["images_name"][0]
#         image1 = result["images_name"][1]
        
#         if len(good_matches) < min_point: # based on opencv documentation, to use RANSAC it needs min. 8 points
#             continue

#         if desc_type == "sift":
#             kps0 = load_sift_output(desc_filepath, image0)[1]
#             kps1 = load_sift_output(desc_filepath, image1)[1]
#         elif desc_type == "disk":
#             disk_output = torch.load(desc_filepath, weights_only=False)
#             features0 = get_disk_output(disk_output, image0)
#             features1 = get_disk_output(disk_output, image1)
        
#             kps0 = convert_result_into_opencv(features0)["keypoints"]
#             kps1 = convert_result_into_opencv(features1)["keypoints"]

#         pts0 = np.float32([kps0[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
#         pts1 = np.float32([kps1[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
#         F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_RANSAC, reproj_thres, confidence)
    
#         if F is None or mask is None:
#             continue
    
#         inlier = [m for i,m in enumerate(good_matches) if mask[i][0] == 1]
#         ransac_result.append({
#             "pair": pair,
#             "images_name" : [image0, image1],
#             "good_matches" : inlier,
#         })

#     return ransac_result