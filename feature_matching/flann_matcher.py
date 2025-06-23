import cv2 as cv
import os
import torch
import sys
sys.path.append(os.path.abspath("./"))
from load_h5py_files import load_sift_output

def convert_result_into_opencv(feature):
    desc = feature[0].descriptors
    desc_np = desc.cpu().detach().numpy().astype("float32")
    kps = feature[0].keypoints
    kps_np = [cv.KeyPoint(x=float(x), y=float(y), size=1) for x,y in kps.cpu().numpy()]
    features_np = {
        "image_name": feature["image_name"],
        "keypoints": kps_np,
        "descriptors": desc_np
    }

    return features_np

def flann_matcher(i, j, descriptors_filepath, images_name=None, descriptor="sift", ratio=0.7):
    '''
        Image matching with FLANN

        Args:
            i, j (int)                  : index image pair
            img_features_filepath (str) : filepath of descriptor output
            images_name (None/list)     : list of images name in a dataset
            descriptor (str)            : descriptor type
            ratio (float)               : percentage for ratio test
        
        Returns:
            list of cv.DMatch           : list of best matches after ratio test
            int, int                    : index image pair
    '''
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    if descriptor == "sift":
        _, _, desc1 = load_sift_output(descriptors_filepath, images_name[i])
        _, _, desc2 = load_sift_output(descriptors_filepath, images_name[j])
    elif descriptor == "disk":
        desc1 = torch.load(descriptors_filepath, weights_only=False)[i]
        desc2 = torch.load(descriptors_filepath, weights_olny=False)[j]

        desc1_np = convert_result_into_opencv(desc1)["descriptors"]
        desc2_np = convert_result_into_opencv(desc2)["descriptors"]

    matches = flann.knnMatch(desc1, desc2, k=2)
    # Ratio test
    good_matches = [m for m,n in matches if m.distance < ratio * n.distance]

    return good_matches, i, j