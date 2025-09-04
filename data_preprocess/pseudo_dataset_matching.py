import torch
from torch.utils.data import IterableDataset
import os
import h5py
import pandas as pd
from typing import Optional, Literal, Dict, List
import torch.nn.functional as F
import math
from torch.utils.data import get_worker_info


def collate_fn(batch: List[Dict]) -> Dict:
    output = {
        'image0': {'keypoints': [], 'descriptors': [], 'image': [], 'keypoints_mask': [], 'scales': [], 'oris': []},
        'image1': {'keypoints': [], 'descriptors': [], 'image': [], 'keypoints_mask': [], 'scales': [], 'oris': []},
        'image0_name': [],
        'image1_name': [],
        'dataset': [],
        'scene0': [],
        'scene1': [],
        'descriptor_type': []
    }

    for item in batch:
        for key in ['image0', 'image1']:
            output[key]['keypoints'].append(item[key]['keypoints'])
            output[key]['descriptors'].append(item[key]['descriptors'])
            output[key]['image'].append(item[key]['image'])
            output[key]['keypoints_mask'].append(item[key].get('keypoints_mask', torch.ones(item[key]['keypoints'].shape[0], dtype=torch.bool)))
            if item['descriptor_type'] == 'sift':
                output[key]['scales'].append(item[key]['scales'])
                output[key]['oris'].append(item[key]['oris'])

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
        if output['descriptor_type'][0] == 'sift':
            output[key]['scales'] = torch.stack(output[key]['scales'])
            output[key]['oris'] = torch.stack(output[key]['oris'])

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
        Returns features dict:
            keypoints: [M x 2]
            descriptors: [M x D]
            image_size: [2]
            keypoints_mask: [M]
            image: [...]
            scene: str
            (optional) scales, oris if descriptor_type=='sift'
        """
        try:
            idx = self.idx_map[(dataset_name, image_name)]
        except KeyError:
            raise KeyError(f"Image not found in merged H5: (dataset={dataset_name}, image={image_name})")
    
        with h5py.File(self.h5_path, 'r') as f:
            keypoints = torch.from_numpy(f['keypoints'][idx])
            descriptors = torch.from_numpy(f['descriptors'][idx])
            image = torch.from_numpy(f['image'][idx])
            mask = torch.from_numpy(f['keypoints_mask'][idx]) if 'keypoints_mask' in f else torch.ones(keypoints.shape[0], dtype=torch.bool)
    
            if 'image_size' in f:
                image_size = torch.from_numpy(f['image_size'][idx])
            else:
                image_size = torch.tensor([1024, 1024], dtype=torch.float32)
    
            features = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image_size': image_size,
                'image': image,
                'scene': f['scene_name'][idx][:].decode('utf-8'),
                'keypoints_mask': mask,
            }
    
            # Optional scores
            if self._has_scores:
                features['keypoint_scores'] = torch.from_numpy(f['keypoint_scores'][idx])
            else:
                features['keypoint_scores'] = torch.zeros(keypoints.shape[0])
    
            # Add SIFT-specific fields if descriptor_type == 'sift'
            if self.descriptor_type == 'sift':
                # Make sure they exist in the H5 file
                if 'scales' in f:
                    features['scales'] = torch.from_numpy(f['scales'][idx])
                else:
                    features['scales'] = torch.ones(keypoints.shape[0], dtype=torch.float32)
    
                if 'oris' in f:
                    features['oris'] = torch.from_numpy(f['oris'][idx])
                else:
                    features['oris'] = torch.zeros(keypoints.shape[0], dtype=torch.float32)
    
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
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            iter_start = 0
            iter_end = len(self.pairs_df)
        else:
            # In multi-worker mode, split workload
            per_worker = int(math.ceil(len(self.pairs_df) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.pairs_df))

        # Yield only the slice assigned to this worker
        for idx in range(iter_start, iter_end):
            row = self.pairs_df.iloc[idx]

            features0 = self._load_features(row['dataset'], row['image1'])
            features1 = self._load_features(row['dataset'], row['image2'])
            

            yield {
                'image0': {
                    'keypoints': features0['keypoints'],
                    'descriptors': features0['descriptors'],
                    'image_size': features0['image_size'],
                    'image': features0['image'],
                    'keypoints_mask': features0.get('keypoints_mask', None),
                    **({'scales': features0['scales'], 'oris': features0['oris']} 
               if self.descriptor_type == 'sift' else {})
                },
                'image1': {
                    'keypoints': features1['keypoints'],
                    'descriptors': features1['descriptors'],
                    'image_size': features1['image_size'],
                    'image': features1['image'],
                    'keypoints_mask': features1.get('keypoints_mask', None),
                    **({'scales': features1['scales'], 'oris': features1['oris']} 
               if self.descriptor_type == 'sift' else {})
                },
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