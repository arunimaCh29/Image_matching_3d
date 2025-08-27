import torch
from torch.utils.data import Dataset
import os
import h5py
import pandas as pd
from typing import Optional, Literal


class PseudoMatchingDataset(Dataset):
    def __init__(
        self,
        pairs_path: str,
        descriptors_path: str,
        descriptor_type: Literal['sift', 'disk'] = 'sift',
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            pairs_path (str): Path to CSV containing image pairs. Must have columns:
                             'image1', 'image2', 'dataset'
            descriptors_path (str): Path to merged H5 containing:
                - keypoints, descriptors, keypoint_scores, keypoints_mask (optional)
                - image_name: list of image names
                - dataset_name: list of dataset names
            descriptor_type (str): Type of descriptor ('sift' or 'disk')
            device: Optional device to move tensors to
        """
        self.pairs_df = pd.read_csv(pairs_path)
        self.h5_path = descriptors_path
        self.descriptor_type = descriptor_type
        self.device = device

        # Build lookup map from (dataset_name, image_name) to index in H5
        with h5py.File(self.h5_path, 'r') as f:
            if 'image_name' not in f or 'dataset_name' not in f:
                raise ValueError("Merged H5 must contain both 'image_name' and 'dataset_name'")
            
            image_names = list(f['image_name'].asstr()[:])
            dataset_names = list(f['dataset_name'].asstr()[:])
            
            # Check if optional keypoint scores/mask exist
            self._has_scores = 'keypoint_scores' in f
            self._has_mask = 'keypoints_mask' in f

        # Create lookup dictionary: (dataset_name, image_name) -> index
        self.idx_map = {(ds, img): idx for idx, (ds, img) in enumerate(zip(dataset_names, image_names))}

    def _move_to_device(self, tensor_dict):
        """Move tensors to specified device if set"""
        if self.device is None:
            return tensor_dict
        
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                for k, v in tensor_dict.items()}

    def _load_features(self, dataset_name: str, image_name: str):
        """Load features for a specific (dataset, image) pair from merged H5"""
        try:
            idx = self.idx_map[(dataset_name, image_name)]
        except KeyError:
            raise KeyError(f"Image not found in merged H5: (dataset={dataset_name}, image={image_name})")

        with h5py.File(self.h5_path, 'r') as f:
            features = {
                'keypoints': torch.from_numpy(f['keypoints'][idx]),
                'descriptors': torch.from_numpy(f['descriptors'][idx]),
                'scene': f['scene_name'][idx][:].decode('utf-8'),
                'image': torch.from_numpy(f['image'][idx]),
            }
            
            if self._has_scores:
                features['keypoint_scores'] = torch.from_numpy(f['keypoint_scores'][idx])
            else:
                features['keypoint_scores'] = torch.zeros(features['keypoints'].shape[0])
                
            if self._has_mask:
                features['keypoints_mask'] = torch.from_numpy(f['keypoints_mask'][idx]).bool()

        return features

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        """Return features for a pair of images"""
        row = self.pairs_df.iloc[idx]
        
        # Load features for both images
        features1 = self._load_features(row['dataset'], row['image1'])
        features2 = self._load_features(row['dataset'], row['image2'])

        return {
            'descriptor_type': self.descriptor_type,
            'dataset': row['dataset'],
            'image1': row['image1'],
            'image2': row['image2'],
            'features1': features1,  # contains keypoints, descriptors, scores, mask, scene
            'features2': features2   # same structure as features1
        }