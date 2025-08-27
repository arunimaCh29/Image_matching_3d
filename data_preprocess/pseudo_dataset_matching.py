import torch
from torch.utils.data import IterableDataset
import os
import h5py
import pandas as pd
from typing import Optional, Literal, Dict, List
import torch.nn.functional as F


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to properly batch the data
    Args:
        batch: List of dictionaries from dataset
    Returns:
        Batched dictionary with proper tensor shapes
    """
    # Initialize output dict with lists
    output = {
        'image0': {'keypoints': [], 'descriptors': [], 'image_size': [], 'keypoints_mask': []},
        'image1': {'keypoints': [], 'descriptors': [], 'image_size': [], 'keypoints_mask': []},
        'image0_name': [],
        'image1_name': [],
        'dataset': [],
        'scene0': [],
        'scene1': [],
        'descriptor_type': []
    }
    
    # Collect items from batch
    for item in batch:
        for key in ['image0', 'image1']:
            output[key]['keypoints'].append(item[key]['keypoints'])
            output[key]['descriptors'].append(item[key]['descriptors'])
            output[key]['image_size'].append(item[key]['image_size'])
            output[key]['keypoints_mask'].append(item[key]['keypoints_mask'] if item[key]['keypoints_mask'] is not None 
                                               else torch.ones_like(item[key]['keypoints'][:, 0], dtype=torch.bool))
        
        output['image0_name'].append(item['image0_name'])
        output['image1_name'].append(item['image1_name'])
        output['dataset'].append(item['dataset'])
        output['scene0'].append(item['scene0'])
        output['scene1'].append(item['scene1'])
        output['descriptor_type'].append(item['descriptor_type'])
    
    # Stack tensors
    for key in ['image0', 'image1']:
        output[key]['keypoints'] = torch.stack(output[key]['keypoints'])
        output[key]['descriptors'] = torch.stack(output[key]['descriptors'])
        output[key]['image_size'] = torch.stack(output[key]['image_size'])
        output[key]['keypoints_mask'] = torch.stack(output[key]['keypoints_mask'])
    
    return output


class PseudoMatchingDataset(IterableDataset):
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
        """
        Load features for a specific (dataset, image) pair from merged H5
        Returns features with shapes:
            keypoints: [M x 2] - becomes [B x M x 2] after batching
            descriptors: [M x D] - becomes [B x M x D] after batching
            image_size: [2] - becomes [B x 2] after batching
        """
        try:
            idx = self.idx_map[(dataset_name, image_name)]
        except KeyError:
            raise KeyError(f"Image not found in merged H5: (dataset={dataset_name}, image={image_name})")

        with h5py.File(self.h5_path, 'r') as f:
            # Load keypoints [M x 2]
            keypoints = torch.from_numpy(f['keypoints'][idx])
            
            # Load descriptors [M x D]
            descriptors = torch.from_numpy(f['descriptors'][idx])
            image = torch.from_numpy(f['image'][idx])
            mask = torch.from_numpy(f['keypoints_mask'][idx])
            
            # Load image size if available, otherwise use default
            if 'image_size' in f:
                image_size = torch.from_numpy(f['image_size'][idx])  # [2]
            else:
                # Default size if not available
                image_size = torch.tensor([1024, 1024], dtype=torch.float32)  # [2]
            
            features = {
                'keypoints': keypoints,  # [M x 2]
                'descriptors': descriptors,  # [M x D]
                'image_size': image_size,  # [2]
                'image': image,
                'scene': f['scene_name'][idx][:].decode('utf-8')
            }
            
            if self._has_scores:
                scores = torch.from_numpy(f['keypoint_scores'][idx])  # [M]
                features['keypoint_scores'] = scores
            else:
                features['keypoint_scores'] = torch.zeros(keypoints.shape[0])  # [M]
                
            if self._has_mask:
                mask = torch.from_numpy(f['keypoints_mask'][idx]).bool()  # [M]
                features['keypoints_mask'] = mask

        return features

    def __iter__(self):
        """
        Iterate over all image pairs
        Returns dict with:
            image0: dict
                keypoints: [M x 2] (becomes [B x M x 2] after batching)
                descriptors: [M x D] (becomes [B x M x D] after batching)
                image_size: [2] (becomes [B x 2] after batching)
                keypoints_mask: [M] boolean mask (becomes [B x M] after batching), or None if no mask
            image1: dict
                keypoints: [N x 2] (becomes [B x N x 2] after batching)
                descriptors: [N x D] (becomes [B x N x D] after batching)
                image_size: [2] (becomes [B x 2] after batching)
                keypoints_mask: [N] boolean mask (becomes [B x N] after batching), or None if no mask
            image0_name: str, filename of first image
            image1_name: str, filename of second image
            dataset: str, name of the dataset
            scene: str, name of the scene
            descriptor_type: str, type of descriptors used ('sift' or 'disk')
        """
        for _, row in self.pairs_df.iterrows():
            # Load features for both images
            features0 = self._load_features(row['dataset'], row['image1'])
            features1 = self._load_features(row['dataset'], row['image2'])

            # Restructure the data to match the required format
            yield {
                'image0': {
                    'keypoints': features0['keypoints'],        # [M x 2]
                    'descriptors': features0['descriptors'],    # [M x D]
                    'image_size': features0['image_size'],      # [2]
                    'image': features0['image'], 
                    'keypoints_mask': features0['keypoints_mask'] if 'keypoints_mask' in features0 else None,  # [M]
                },
                'image1': {
                    'keypoints': features1['keypoints'],        # [N x 2]
                    'descriptors': features1['descriptors'],    # [N x D]
                    'image_size': features1['image_size'],      # [2]
                    'image': features1['image'], 
                    'keypoints_mask': features1['keypoints_mask'] if 'keypoints_mask' in features1 else None,  # [N]
                },
                # Image names and metadata
                'image0_name': row['image1'],
                'image1_name': row['image2'],
                'dataset': row['dataset'],
                'scene0': features0['scene'],
                'scene1': features1['scene'],
                'descriptor_type': self.descriptor_type
            }
            
    def __len__(self):
        """Optional: Keep len() support for progress tracking"""
        return len(self.pairs_df)