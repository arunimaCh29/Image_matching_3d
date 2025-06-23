import kornia
from kornia.feature import DISK

def get_disk_outputs(dataset, dataset_indices, device):
    disk = DISK.from_pretrained("depth").to(device)
    features = []
    for i, idx in enumerate(dataset_indices):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        feature = disk(img, pad_if_not_divisible=True)
        features.append({
            "image_name": sample["image_name"],
            "feature": feature
        })
    
    return features