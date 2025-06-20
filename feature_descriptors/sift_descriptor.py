import numpy as np
import torch
import cv2 as cv

def get_keypoint_and_descriptor(image, sift_nfeatures=None):
    """
    Get the keypoint and descriptor of the image

    Args:
        image (tensor or numpy.ndarray) : input image, should be at least as tensor (but doesn't need to be normalized ) or numpy array
        sift_nfeatures : maximum features for SIFT
    
    Returns:
        numpy.ndarray   : image in grayscale as numpy array
        tuple           : tuple of keypoints
        numpy.ndarray   : array of arrays represented the feature descriptors
    """

    # Check if image is transformed as tensor
    if torch.is_tensor(image):
        # Adjust the tensor shape to make it suitable in OpenCV
        img = torch.permute(image, (1,2,0))
        # Convert tensor to numpy, OpenCV SIFT needs the input as numpy array
        img_np = img.numpy()
        img_np = img_np.astype(np.uint8)
    else:
        img_np = image
    
    # Convert RGB to grayscale
    if len(img_np.shape) > 2 and img_np.shape[2] == 3:
        img_np = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)

    # Create SIFT
    sift = cv.SIFT_create(nfeatures=sift_nfeatures) if sift_nfeatures else cv.SIFT_create()
    # Compute the image's keypoint and descriptor
    keypoint, descriptor = sift.detectAndCompute(img_np, None)

    return img_np, keypoint, descriptor

