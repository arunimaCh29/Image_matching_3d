import torch
from torch.utils.data import Dataset
import os
import h5py  # for reading descriptor files
import pandas as pd

class PseudoMatchingDataset(Dataset):
    def __init__(self, labels_path, descriptors_dir, descriptor_type='sift'):
        """
        Args:
            labels_path (str): Path to CSV containing image pairs and scene info
            descriptors_dir (str): Directory containing descriptor files
            descriptor_type (str): Type of descriptor ('sift' or 'disk')
        """
        self.labels_df = pd.read_csv(labels_path)
        self.descriptors_dir = descriptors_dir
        self.descriptor_type = descriptor_type
        
        # Create pairs list with all necessary information
        self.pairs = self._create_pairs()
        
    def _create_pairs(self):
        """Create list of pairs with their metadata and load descriptors"""
        pairs = []
        
        for _, row in self.labels_df.iterrows():
            # Assuming your CSV has columns: image1, image2, dataset, scene
            img1_name = row['image1']
            img2_name = row['image2']
            dataset_name = row['dataset']
            scene_name = row['scene']
            
            # Load descriptors for both images
            img1_desc = self._load_descriptor(dataset_name, img1_name)
            img2_desc = self._load_descriptor(dataset_name, img2_name)
            
            # Create pair dictionary with all information
            pair_dict = {
                # Original metadata
                'dataset': dataset_name,
                'scene': scene_name,
                'image1_name': img1_name,
                'image2_name': img2_name,
                
                'keypoints1': img1_desc['keypoints'],
                'keypoint_scores1': img1_desc['keypoint_scores'],
                'descriptors1': img1_desc['descriptors'],
                'image_size1': img1_desc['image_size'],
                
                'keypoints2': img2_desc['keypoints'],
                'keypoint_scores2': img2_desc['keypoint_scores'],
                'descriptors2': img2_desc['descriptors'],
                'image_size2': img2_desc['image_size']
            }
            
            pairs.append(pair_dict)
            
        return pairs
    
    def _load_descriptor(self, dataset_name, image_name):
        """Load descriptor file for a specific image"""
        desc_file = os.path.join(
            self.descriptors_dir,
            f"{dataset_name}_{image_name}_{self.descriptor_type}.h5"
        )
        
        with h5py.File(desc_file, 'r') as f:
            descriptor_dict = {
                'keypoints': torch.from_numpy(f['keypoints'][()]),
                'keypoint_scores': torch.from_numpy(f['keypoint_scores'][()]),
                'descriptors': torch.from_numpy(f['descriptors'][()]),
                'image_size': torch.from_numpy(f['image_size'][()])
            }
        
        return descriptor_dict
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Return a pair of descriptors with all metadata"""
        return self.pairs[idx]

