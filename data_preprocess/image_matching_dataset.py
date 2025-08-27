import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Dict

class ImageMatchingDataset(Dataset):
    def __init__(self, labels_path: str, root_dir: str, transform=None):
        """
        Custom Dataset for image matching
        
        Args:
            labels_path (str): Path to the labels CSV file
            root_dir (str): Root directory containing the dataset folders
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        
        # Create a list of valid image paths
        self.image_paths = []
        self.valid_indices = []
        
        for idx, row in self.labels_df.iterrows():
            img_path = self.root_dir / row['dataset'] / row['image']
            if img_path.exists():
                self.image_paths.append(img_path)
                self.valid_indices.append(idx)
        
        # Filter the dataframe to only include existing images
        self.labels_df = self.labels_df.iloc[self.valid_indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_matrix(self, matrix_str: str) -> torch.Tensor:
        """Parse semicolon-separated matrix string into tensor"""
        if pd.isna(matrix_str):
            return None
        values = [float(x) for x in matrix_str.split(';')]
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing image and its metadata
        """
        # Get image path and load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        width, height = image.size
        image_size = torch.tensor([width, height], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        # Get corresponding label row
        label_row = self.labels_df.iloc[idx]
        
        # Parse rotation matrix and translation vector
        rotation_matrix = self._parse_matrix(label_row['rotation_matrix'])
        translation_vector = self._parse_matrix(label_row['translation_vector'])
        
        if rotation_matrix is not None:
            rotation_matrix = rotation_matrix.reshape(3, 3)
        
        if translation_vector is not None:
            translation_vector = translation_vector.reshape(3, 1)


            
        item = {
            'image': image,
            'dataset_name': label_row['dataset'],
            'scene_name': label_row['scene'],
            'image_name': label_row['image'],
            'image_path': str(img_path),
            'image_size': image_size,
            'rotation_matrix': rotation_matrix,
            'translation_vector': translation_vector
        }
        
        return item