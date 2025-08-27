import h5py
import cv2 as cv
import numpy as np
import torch
from pathlib import Path

def load_sift_output(filepath, img_name):
    with h5py.File(filepath, "r") as f:
        group = f[img_name]
        kps_arr = group["keypoints"][:]
        desc = group["descriptors"][:]
        img_np = group["image_np"][:]

        kps = [
            cv.KeyPoint(
                x=row[0], y=row[1],
                size=row[2],
                angle=row[3],
                response=row[4],
                octave=int(row[5]),
                class_id=int(row[6])
            )
            for row in kps_arr
        ]

        return img_np, kps, desc

def load_flann_output(filepath, pair_idx):
    with h5py.File(filepath,"r") as f:
        group = f[pair_idx]
        matches_arr = group["good_matches"][:]
        pair = group["pair"][:]
        images_name = group["images_name"].asstr()[:]

        matches = [
            cv.DMatch(
                _queryIdx=int(row[0]),
                _trainIdx=int(row[1]),
                _imgIdx=int(row[2]),
                _distance=row[3]
            )
            for row in matches_arr
        ]

        return {"pair": tuple(pair), "images_name": images_name, "good_matches" : matches}

def load_flann_from_images_name(filepath, images_name):
    with h5py.File(filepath, "r") as f:
        for idx in f.keys():
            group = f[idx]
            img_name = group["images_name"].asstr()[:]
            # if (img_name != images_name).all():
            if not np.array_equal(np.sort(img_name), np.sort(np.array(images_name))):
                continue

            data = load_flann_output(filepath, idx)
            break
        
        return data


def save_matches_to_h5(matches_list, output_path, matcher_type):
    """
    Save matches to H5 file with masks
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with h5py.File(output_path, 'w') as f:
        #f.attrs['matcher'] = matcher_type
        dt = h5py.string_dtype(encoding='utf-8')
        
        for idx, match_data in enumerate(matches_list):
            group = f.create_group(str(idx))
            
            # Store basic info
            group.create_dataset('image1', data=match_data['image1'], dtype=dt)
            group.create_dataset('image2', data=match_data['image2'], dtype=dt)
            group.create_dataset('dataset', data=match_data['dataset'], dtype=dt)
            
            # Store scenes
            scene1 = match_data['features1'].get('scene', 'none')
            scene2 = match_data['features2'].get('scene', 'none')
            if scene1 is None: scene1 = 'none'
            if scene2 is None: scene2 = 'none'
            group.create_dataset('scene1', data=scene1, dtype=dt)
            group.create_dataset('scene2', data=scene2, dtype=dt)
            
            # Store keypoint coordinates and matches
            group.create_dataset('points0', data=match_data['points0'].numpy())
            group.create_dataset('points1', data=match_data['points1'].numpy())
            
            # Store masks if present
            if 'keypoints_mask' in match_data['features1']:
                group.create_dataset('mask1', data=match_data['features1']['keypoints_mask'].numpy())
            if 'keypoints_mask' in match_data['features2']:
                group.create_dataset('mask2', data=match_data['features2']['keypoints_mask'].numpy())
            
            if matcher_type == 'lightglue':
                group.create_dataset('matches', data=match_data['matches'])
            elif matcher_type == 'flann':
                group.create_dataset('matches_idx', data=match_data['matches_idx'].numpy())
                group.create_dataset('distances', data=match_data['distances'].numpy())

def load_matches_from_h5(h5_path):
    """
    Load matches from H5 file including masks
    """
    matches_list = []
    
    with h5py.File(h5_path, 'r') as f:
        matcher_type = 'lightglue'
        
        for idx in f.keys():
            group = f[idx]
            print(f[idx])
            
            # Load basic data
            match_data = {
                'image1': group['image1'][()].decode('utf-8'),
                'image2': group['image2'][()].decode('utf-8'),
                'dataset': group['dataset'][()].decode('utf-8'),
                'points0': torch.from_numpy(group['points0'][()]),
                'points1': torch.from_numpy(group['points1'][()])
            }
            
            # Load scenes
            scene1 = group['scene1'][()].decode('utf-8')
            scene2 = group['scene2'][()].decode('utf-8')
            match_data['features1'] = {'scene': None if scene1 == 'none' else scene1}
            match_data['features2'] = {'scene': None if scene2 == 'none' else scene2}
            
            # Load masks if present
            if 'mask1' in group:
                match_data['features1']['keypoints_mask'] = torch.from_numpy(group['mask1'][()])
            if 'mask2' in group:
                match_data['features2']['keypoints_mask'] = torch.from_numpy(group['mask2'][()])
            
            # Load matcher-specific data
            if matcher_type == 'lightglue':
                match_data['matches'] ='d'# group['matches']
            else:  # flann
                match_data['matches_idx'] = torch.from_numpy(group['matches_idx'][()])
                match_data['distances'] = torch.from_numpy(group['distances'][()])
            
            matches_list.append(match_data)
    
    return matches_list