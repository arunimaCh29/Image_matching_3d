from feature_matching.LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from feature_matching.LightGlue.lightglue.utils import load_image, rbd
from typing import Literal

def get_SIFT_features(image_location, cuda=False, max_keypoints=2048) -> dict:
    if cuda:
        extractor = SIFT(max_num_keypoints=max_keypoints).eval().cuda()  # load the extractor
        image = load_image(image_location).cuda()
    else:
        extractor = SIFT(max_num_keypoints=max_keypoints).eval()  # load the extractor
        image = load_image(image_location)

    feats = extractor.extract(image)  # auto-resize the image, disable with resize=None

    return feats

def get_DISK_features(image_location, cuda=False, max_keypoints=2048) -> dict:
    if cuda:
        extractor = DISK(max_num_keypoints=max_keypoints).eval().cuda()  # load the extractor
        image = load_image(image_location).cuda()
    else:
        extractor = DISK(max_num_keypoints=max_keypoints).eval()  # load the extractor
        image = load_image(image_location)

    feats = extractor.extract(image)  # auto-resize the image, disable with resize=None

    return feats


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

