import pandas as pd
import numpy as np
import ast
import sys
import os
sys.path.append(os.path.abspath("./"))
from load_h5py_files import load_matches_from_h5

class MatcherCsv():
    def __init__(self, matcher_type, descriptor_type, matcher_dir, csv_output_dir):
        '''
        Args:
            matcher_type (str): type of matcher (flann / lightglue)
            descriptor_type (str): type of descriptor (sift / disk)
            matcher_dir (str): path to matcher result
            csv_output_dir (str): path to save the merged image matching result as CSV
        '''
        self.matcher_type = matcher_type
        self.descriptor_type = descriptor_type
        self.matcher_dir = matcher_dir
        self.csv_output_dir = csv_output_dir
        self.matcher_data = self.create_matcher_csv()

    def create_matcher_csv(self):
        '''
        Merge image matching result in one CSV
        '''
        all_data = []

        for file in os.listdir(self.matcher_dir):
            if not file.endswith(".h5"): # skip .ipynb_checkpoints folder
                continue

            filepath = os.path.join(self.matcher_dir, file)
            # load image matching result from .h5 file
            file_data = load_matches_from_h5(filepath)

            # convert tensor data into list
            for data in file_data:
                data["points0"] = data["points0"].cpu().detach().numpy().tolist()
                data["points1"] = data["points1"].cpu().detach().numpy().tolist()

                # convert mask into list if present
                if "mask1" in data["features1"].keys():
                    data["features1"]["mask1"] = data["features1"]["mask1"].cpu().detach().numpy().tolist()
                if "mask2" in data["features2"].keys():
                    data["features2"]["mask2"] = data["features2"]["mask2"].cpu().detach().numpy().tolist()

                # convert matcher-specific data into list
                if self.matcher_type.lower() == "lightglue":
                    data["matches"] = data["matches"].cpu().detach().numpy().tolist()
                elif self.matcher_type.lower() == "flann":
                    data["matches_idx"] = data["matches_idx"].cpu().detach().numpy().tolist()
                    data["distances"] = data["distances"].cpu().detach().numpy().tolist()

            # merge data
            all_data += file_data

        df_all_data = pd.DataFrame(all_data)

        os.makedirs(self.csv_output_dir, exist_ok=True)
        csv_filepath = os.path.join(self.csv_output_dir, f"result_{self.matcher_type}_{self.descriptor_type}.csv")

        df_all_data.to_csv(csv_filepath, index=False)