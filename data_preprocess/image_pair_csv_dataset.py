import pandas as pd
import os
from itertools import combinations

class ImagePairCsv():
    def __init__(self, labels_path, output_dir):
        '''
        Args:
            labels_path (str): path to train labels CSV
            output_dir (str): path to save the image pairing data
        '''
        self.labels_df = pd.read_csv(labels_path)
        self.output_dir = output_dir

        self.pairs_csv = self.create_csv()
    
    def create_csv(self):
        '''
        Create image pairing between images of same dataset, save as CSV
        '''
        dataset = self.labels_df["dataset"].unique()
        first_images, second_images, datasets = [], [], []

        for dataset_name in dataset:
            dataset_df = self.labels_df[self.labels_df["dataset"] == dataset_name]
            image_combinations = list(combinations(dataset_df["image"].unique(), 2))
            first_pair_images, second_pair_images = zip(*image_combinations)
            first_pair_images = list(first_pair_images)
            second_pair_images = list(second_pair_images)

            first_images += first_pair_images
            second_images += second_pair_images
            datasets += ([dataset_name] * len(first_pair_images))

            # pair_data = {
            #     "image1" : first_pair_images,
            #     "image2" : second_pair_images,
            #     "dataset" : dataset_name,
            # }

        pair_data = {
            "image1" : first_images,
            "image2" : second_images,
            "dataset" : datasets
        }

        pair_df = pd.DataFrame(pair_data)

        os.makedirs(self.output_dir, exist_ok=True)

        file_path = os.path.join(self.output_dir, "data_pairs.csv")
        pair_df.to_csv(file_path, index=False)