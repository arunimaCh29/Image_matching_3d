import h5py
import cv2 as cv
import numpy as np

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