from typing import Literal
import pandas as pd
from feature_matching.lightglue_matcher import get_SIFT_features, get_DISK_features, match_features







def batch_match_lightglue(image_i: str, image_j: str, descriptor: Literal['disk', 'sift'], cuda=False, max_keypoints=2048):

    if descriptor == 'sift':
        feats0 = get_SIFT_features(image_i, cuda=cuda, max_keypoints=max_keypoints)
        feats1 = get_SIFT_features(image_j, cuda=cuda, max_keypoints=max_keypoints)
    else:
        feats0 = get_DISK_features(image_i, cuda=cuda, max_keypoints=max_keypoints)
        feats1 = get_DISK_features(image_j, cuda=cuda, max_keypoints=max_keypoints)

    points = match_features(image0_features=feats0, image1_features=feats1, descriptor=descriptor, cuda=cuda)

    return points


def batch_match_flann(image_i: str, image_j: str, descriptor: Literal['disk', 'sift'], cuda=False, max_keypoints=2048):
    return 0



def batch_match_features(dataset: pd.DataFrame, matcher: Literal['lightglue', 'flann'], descriptor: Literal['disk', 'sift'], cuda=False, max_keypoints=2048):
    
    matcher_matrix = {}
    for index_i, image_i in dataset.iterrows():
        matcher_matrix.update({image_i['image']: {}})


    for index_i, image_i in dataset.iterrows():
        for index_j, image_j in dataset.iterrows():
            if image_i['image'] == image_j['image']:
                continue
            if image_j['image'] in matcher_matrix[image_i['image']]:
                continue
            if matcher == 'lightglue':
                matcher_matrix[image_i['image']].update({image_j['image']: batch_match_lightglue(image_i['image'], image_j['image'], descriptor, cuda=cuda, max_keypoints=max_keypoints)})
                matcher_matrix[image_j['image']].update({image_i['image']: matcher_matrix[image_i['image']][image_j['image']]})
            if matcher == 'flann':
                matcher_matrix[image_i['image']].update({image_j['image']: batch_match_flann(image_i['image'], image_j['image'], descriptor, cuda=cuda, max_keypoints=max_keypoints)})
                matcher_matrix[image_j['image']].update({image_i['image']: matcher_matrix[image_i['image']][image_j['image']]})



