from typing import Literal
import pandas as pd
from feature_matching.lightglue_matcher import get_SIFT_features, get_DISK_features, match_features
import h5py
from pathlib import Path
from feature_matching.lightglue_matcher import match_features_for_plots
from load_h5py_files import save_matches_to_h5

import torch
from kornia.feature import LightGlue
import os
from tqdm import tqdm


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





def process_batches(loader, output_path=None, device=None):
    all_matches = []
    total_batches = len(loader)
     
    print(f"Processing {total_batches} batches...")
    

    for batch_idx, batch in enumerate(tqdm(loader, desc="Matches for the loader", unit="batch")):

        #print(f"Batch {batch_idx+1}/{total_batches}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()


        try:
            descriptor = batch['descriptor_type'][0]
#                matcher = LightGlue(features=descriptor.lower()).eval().to(device)
            batch_matches = []
            
            # Process each item in batch separately
            for i in range(len(batch['image0_name'])):
                with torch.inference_mode():

                    # Get valid masks for unpadding
                    valid_mask0 = batch['image0'].get('keypoints_mask', None)
                    valid_mask1 = batch['image1'].get('keypoints_mask', None)

                    feats0_item = {
                        'keypoints': batch['image0']['keypoints'][i:i+1],
                        'descriptors': batch['image0']['descriptors'][i:i+1],
                        'image': batch['image0']['image'][i:i+1],
                        'scene': batch['scene0'][i]
                    }
                    
  
                    feats1_item = {
                        'keypoints': batch['image1']['keypoints'][i:i+1],
                        'descriptors': batch['image1']['descriptors'][i:i+1],
                        'image': batch['image1']['image'][i:i+1],
                        'scene': batch['scene1'][i]
                    }

                    if descriptor.lower() == 'sift':
                        feats0_item['scales'] = batch['image0']['scales'][i]
                        feats0_item['oris'] = batch['image0']['oris'][i]
                        feats1_item['scales'] = batch['image1']['scales'][i]
                        feats1_item['oris'] = batch['image1']['oris'][i]

                    
                                        
                    # Unpad using valid masks if they exist
                    if valid_mask0 is not None:
                        mask0_i = valid_mask0[i]
                        feats0_item['keypoints'] = feats0_item['keypoints'][:, mask0_i]
                        feats0_item['descriptors'] = feats0_item['descriptors'][:, mask0_i]
                        if 'keypoint_scores' in batch['image0']:
                            feats0_item['keypoint_scores'] = batch['image0']['keypoint_scores'][i:i+1, mask0_i]
                        if descriptor.lower() == 'sift':
                            feats0_item['scales'] = feats0_item['scales'][mask0_i].unsqueeze(0) if feats0_item['scales'] is not None else None
                            feats0_item['oris'] = feats0_item['oris'][mask0_i].unsqueeze(0) if feats0_item['oris'] is not None else None

                    
                    if valid_mask1 is not None:
                        mask1_i = valid_mask1[i]  # Get mask for current item
                        feats1_item['keypoints'] = feats1_item['keypoints'][:, mask1_i]
                        feats1_item['descriptors'] = feats1_item['descriptors'][:, mask1_i]
                        if 'keypoint_scores' in batch['image1']:
                            feats1_item['keypoint_scores'] = batch['image1']['keypoint_scores'][i:i+1, mask1_i]
                        if descriptor.lower() == 'sift':
                            feats1_item['scales'] = feats1_item['scales'][mask1_i].unsqueeze(0) if feats1_item['scales'] is not None else None
                            feats1_item['oris'] = feats1_item['oris'][mask1_i].unsqueeze(0) if feats1_item['oris'] is not None else None

                    
                    
       
                    # Move features to device
                    feats0_device = {k: v.to(device) if torch.is_tensor(v) and v is not None else v 
                                   for k, v in feats0_item.items()}
                    feats1_device = {k: v.to(device) if torch.is_tensor(v) and v is not None else v 
                                   for k, v in feats1_item.items()}
                    
                    # Run matching for current item
                    points0, points1, matches = match_features_for_plots(feats0_device,feats1_device,descriptor=descriptor, device= device,cuda=True)
                    
                    # Create match result for current item
                    match_result = {
                        'image1': batch['image0_name'][i],
                        'image2': batch['image1_name'][i],
                        'features1': feats0_item,
                        'features2': feats1_item,
                        'dataset': batch['dataset'][i],
                        'matches': matches,  # Move to CPU
                        'points0': points0.cpu(),  # Move to CPU
                        'points1': points1.cpu(),  # Move to CPU
                    }
                    batch_matches.append(match_result)
                    
                    # Clean up GPU memory
                    del feats0_device, feats1_device, matches, points0, points1
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            out_path = os.path.join(output_path, f"{batch_idx}_lightglue_{descriptor}.h5")
            save_matches_to_h5(batch_matches,out_path,'LightGlue')
            del batch_matches
            
        except RuntimeError as e:
            # Clear any partial results that might be in memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if "out of memory" in str(e):
                print(f"WARNING: Out of memory in batch {batch_idx+1}. Skipping this batch.")
                continue
            raise e
