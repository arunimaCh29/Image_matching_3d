from typing import Literal
import pandas as pd
from feature_matching.lightglue_matcher import get_SIFT_features, get_DISK_features, match_features
import h5py
from pathlib import Path
from feature_matching.lightglue_matcher import match_features_for_plots
import cv2 as cv
import numpy as np
from load_h5py_files import save_matches_to_h5
import torch




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





def process_batches(loader, matcher='lightglue', output_path=None):
    all_matches = []
    total_batches = len(loader)
    
    # FLANN parameters if needed
    if matcher == 'flann':
        FLANN_INDEX_KDTREE = 1
        flann = cv.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=50)
        )

    print(f"Processing {total_batches} batches with {matcher}...")
    
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx+1}/{total_batches}")
        
        batch_matches = []
        for i in range(len(batch['image1'])):
            descriptor_type = batch["descriptor_type"][i]
            if matcher == 'lightglue':
                # Prepare features for LightGlue (already on correct device from dataset)
                feats0 = {
                    'keypoints': batch['features1']['keypoints'][i].unsqueeze(0),
                    'keypoint_scores': batch['features1']['keypoint_scores'][i].unsqueeze(0),
                    'descriptors': batch['features1']['descriptors'][i].unsqueeze(0)
                }
                feats1 = {
                    'keypoints': batch['features2']['keypoints'][i].unsqueeze(0),
                    'keypoint_scores': batch['features2']['keypoint_scores'][i].unsqueeze(0),
                    'descriptors': batch['features2']['descriptors'][i].unsqueeze(0)
                }

                # Add masks if present
                if 'keypoints_mask' in batch['features1']:
                    feats0['mask'] = batch['features1']['keypoints_mask'][i].unsqueeze(0)
                if 'keypoints_mask' in batch['features2']:
                    feats1['mask'] = batch['features2']['keypoints_mask'][i].unsqueeze(0)

                # Run LightGlue matching
                points0, points1, matches01 = match_features_for_plots(
                    feats0, feats1,
                    descriptor=batch['descriptor_type'][0], # Since all descriptors will be of same type 
                    cuda=torch.cuda.is_available()
                )

                match_result = {
                    'image1': batch['image1'][i],
                    'image2': batch['image2'][i],
                    'dataset': batch['dataset'][i],
                    'matches': matches01['matches'],
                    'points0': points0,
                    'points1': points1,
                    'features1': {'scene': batch['features1']['scene'][i]},
                    'features2': {'scene': batch['features2']['scene'][i]}
                }

            else:  # FLANN
                # Convert to numpy for OpenCV
                desc1 = batch['features1']['descriptors'][i].cpu().numpy().astype('float32')
                desc2 = batch['features2']['descriptors'][i].cpu().numpy().astype('float32')
                
                # Apply masks if present
                if 'keypoints_mask' in batch['features1']:
                    mask1 = batch['features1']['keypoints_mask'][i].cpu().numpy()
                    mask2 = batch['features2']['keypoints_mask'][i].cpu().numpy()
                    desc1 = desc1[mask1]
                    desc2 = desc2[mask2]

                # Run FLANN matching
                matches = flann.knnMatch(desc1, desc2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

                # Get matched points
                if good:
                    matches_idx = torch.tensor([[m.queryIdx, m.trainIdx] for m in good])
                    distances = torch.tensor([m.distance for m in good])
                    kpts1 = batch['features1']['keypoints'][i].cpu()
                    kpts2 = batch['features2']['keypoints'][i].cpu()
                    points0 = kpts1[matches_idx[:, 0]]
                    points1 = kpts2[matches_idx[:, 1]]
                else:
                    matches_idx = torch.empty((0, 2), dtype=torch.long)
                    distances = torch.empty(0)
                    points0 = torch.empty((0, 2))
                    points1 = torch.empty((0, 2))

                match_result = {
                    'image1': batch['image1'][i],
                    'image2': batch['image2'][i],
                    'dataset': batch['dataset'][i],
                    'matches_idx': matches_idx,
                    'distances': distances,
                    'points0': points0,
                    'points1': points1,
                    'features1': {'scene': batch['features1']['scene'][i]},
                    'features2': {'scene': batch['features2']['scene'][i]}
                }

            batch_matches.append(match_result)
        
        all_matches.extend(batch_matches)
        
        # Optional: Save periodically
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"Saving checkpoint after batch {batch_idx+1}")
            filename = output_path / f"{batch_idx+1}_{matcher}_{descriptor_type}.h5"
            save_matches_to_h5(all_matches, filename, matcher)
            all_matches = []
    
    # Final save
    # save_matches_to_h5(all_matches, output_path, matcher)
    return all_matches
