from typing import Literal
import pandas as pd
from feature_matching.lightglue_matcher import get_SIFT_features, get_DISK_features, match_features




feats0 = get_SIFT_features('.\\data\\train\\amy_gardens\\peach_0004.png')
feats1 = get_SIFT_features('.\\data\\train\\amy_gardens\\peach_0008.png')
points0, points1 = match_features(image0_features=feats0, image1_features=feats1, descriptor='sift')

print(len(points0))
print(len(points1))



def batch_match_lightglue(image_i: str, image_j: str, descriptor: Literal['disk', 'sift'], cuda=False)

def batch_match_features(dataset: pd.DataFrame, matcher: Literal['lightglue', 'flann'], descriptor: Literal['disk', 'sift'], cuda=False):
    
    
    
    for index_i, image_i in dataset.iterrows():
        for index_j, image_j in dataset.iterrows():
            if image_i == image_j:
                continue

