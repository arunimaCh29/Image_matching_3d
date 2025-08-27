from feature_matching.LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from feature_matching.LightGlue.lightglue.utils import load_image, rbd
from typing import Literal
import pandas as pd
import torch

def get_SIFT_features(image_location, cuda=False, max_keypoints=2048) -> dict:
    if cuda:
        extractor = SIFT(max_num_keypoints=max_keypoints).eval().cuda()  # load the extractor
        image = load_image(image_location).cuda()
    else:
        extractor = SIFT(max_num_keypoints=max_keypoints).eval()  # load the extractor
        image = load_image(image_location)

    feats = extractor.extract(image)  # auto-resize the image, disable with resize=None

    return feats

def get_DISK_features(image, cuda=False, max_keypoints=2048, device= None) -> dict:
    if device:
        extractor = DISK(max_num_keypoints=max_keypoints).eval().to(device)  # load the extractor
        #image = load_image(image_location).cuda()
    else:
        extractor = DISK(max_num_keypoints=max_keypoints).eval()  # load the extractor
        #image = load_image(image_location)

    feats = extractor({'image':image})  # auto-resize the image, disable with resize=None

    return feats


def match_features_for_plots(image0_features: dict, image1_features: dict, descriptor: Literal['disk', 'sift', 'aliked', 'superpoint', 'doghardnet'], cuda=False, device=None):
    if device:
        matcher = LightGlue(features=descriptor.lower()).eval().to(device)  # load the matcher
        #print(device)
    else:
        matcher = LightGlue(features=descriptor.lower()).eval()  # load the matcher

    matches01 = matcher({'image0': image0_features, 'image1': image1_features})
    image0_features, image1_features, matches01 = [rbd(x) for x in [image0_features, image1_features, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = image0_features['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = image1_features['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

    return points0, points1, matches01

def match_features(image0_features: dict, image1_features: dict, descriptor: Literal['disk', 'sift', 'aliked', 'superpoint', 'doghardnet'], cuda=False):
    if cuda:
        matcher = LightGlue(features=descriptor.lower()).eval().cuda()  # load the matcher
    else:
        matcher = LightGlue(features=descriptor.lower()).eval()  # load the matcher

    matches01 = matcher({'image0': image0_features, 'image1': image1_features})
    image0_features, image1_features, matches01 = [rbd(x) for x in [image0_features, image1_features, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = image0_features['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = image1_features['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

    return points0, points1


def match_features_for_plots_batch(image0_features: dict, image1_features: dict, descriptor: Literal['disk', 'sift', 'aliked', 'superpoint', 'doghardnet'], cuda=False, device=None):
    """
    Match features between batches of image pairs using LightGlue
    
    Args:
        image0_features (dict): Batch of features from first images with shape:
            - keypoints: [B x M x 2]
            - descriptors: [B x M x D]
            - image_size: [B x 2]
        image1_features (dict): Batch of features from second images with same structure
        descriptor (str): Type of descriptor used
        cuda (bool): Whether to use CUDA
        device: Specific device to use
    """
    if device is None:
        device = torch.device('cuda' if cuda else 'cpu')
    
    # Clear CUDA cache before processing
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Move features to device
    feats0 = {
        'keypoints': image0_features['keypoints'].to(device),
        'descriptors': image0_features['descriptors'].to(device),
        'image_size': image0_features['image_size'].to(device)
    }
    feats1 = {
        'keypoints': image1_features['keypoints'].to(device),
        'descriptors': image1_features['descriptors'].to(device),
        'image_size': image1_features['image_size'].to(device)
    }
    
    # Create and move matcher to device
    matcher = LightGlue(features=descriptor.lower()).eval().to(device)
    
    # Process entire batch
    with torch.cuda.amp.autocast():
        matches01 = matcher({'image0': feats0, 'image1': feats1})
    
    # Extract matches and points for each pair in batch
    batch_size = feats0['keypoints'].shape[0]
    all_points0, all_points1, all_matches = [], [], []
    
    for b in range(batch_size):
        matches = matches01['matches'][b]  # Get matches for this item
        points0 = feats0['keypoints'][b][matches[..., 0]]  # [K x 2]
        points1 = feats1['keypoints'][b][matches[..., 1]]  # [K x 2]
        
        # Move results to CPU
        all_points0.append(points0.cpu())
        all_points1.append(points1.cpu())
        all_matches.append(matches.cpu())
    
    # Clean up
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Combine results
    matches01 = {'matches': all_matches}
    
    return torch.stack(all_points0), torch.stack(all_points1), matches01

