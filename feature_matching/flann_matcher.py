import cv2 as cv
import os
import sys
sys.path.append(os.path.abspath("./"))
from load_h5py_files import load_sift_output

def flann_matcher(i, j, descriptors_filepath, images_name, descriptor="SIFT", ratio=0.7):
    '''
        Image matching with FLANN

        Args:
            i, j (int)                  : index image pair
            img_features_filepath (str) : filepath of descriptor output
            images_name (list)          : list of images name in a dataset
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
    if descriptor == "SIFT":
        _, _, desc1 = load_sift_output(descriptors_filepath, images_name[i])
        _, _, desc2 = load_sift_output(descriptors_filepath, images_name[j])

    matches = flann.knnMatch(desc1, desc2, k=2)
    # Ratio test
    good_matches = [m for m,n in matches if m.distance < ratio * n.distance]

    return good_matches, i, j