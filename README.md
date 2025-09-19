# Image Matching and Clustering

This project focuses on evaluating different feature descriptors (SIFT, DISK) and matching algorithms (LightGlue, FLANN) for image matching and clustering tasks. It provides tools for:

- **Feature Extraction:** Extracting keypoints and descriptors from images.
- **Feature Matching:** Matching features between image pairs.
- **Clustering:** Grouping images based on their visual similarity derived from feature matches.
- **Evaluation:** Comprehensive evaluation of clustering results using various metrics.

## Project Structure

- `batch_descriptor.py`: Contains functions for extracting SIFT and DISK features from images and saving them in HDF5 format.
- `batch_matcher.py`: Implements feature matching using LightGlue and FLANN, saving the matches in HDF5 format.
- `evaluation/eval.py`: Provides a comprehensive suite of functions for evaluating image clustering performance, including metrics like Homogeneity, Completeness, V-Measure, and detailed cluster-scene relationship analysis.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `feature_descriptors/`: Contains modules for specific descriptor implementations.
- `feature_matching/`: Contains modules for specific matcher implementations, including a LightGlue submodule.
- `data_preprocess/`: Modules for handling image datasets and preparing data for matching.
- `clustering/`: Modules related to image clustering algorithms.
- `evaluation/`: Contains scripts and outputs related to evaluation, including cluster visualizations, graphs, and match outputs.
- `plots/`: Stores various plots generated during evaluation.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arunimaCh29/Image_matching_3d.git
   cd Image_matching_3d
   ```

2. **Download Data:**
   Due to the large size of the datasets, It is not included in the repository. Please download the necessary dataset from [Image_Matching_Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025/data)  and place them in the `data/` directory. Ensure the directory structure within `data/` matches what is expected by the data loading scripts (e.g., `data_preprocess/image_matching_dataset.py`).

3. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Note: Some dependencies like `torch` might require specific installation instructions based on your CUDA version. Refer to the official PyTorch documentation for details.

5. **Initialize LightGlue submodule:**
   ```bash
   git submodule update --init --recursive
   ```

## Usage

The project typically involves a pipeline of feature extraction, matching, and clustering, followed by evaluation.

### 1. Feature Extraction

To extract features (e.g., DISK or SIFT) from your images, you can use the `batch_descriptor.py` script. You will need to provide a data loader and specify the descriptor type and an output directory.

Example:
```python
from data_preprocess.image_matching_dataset import ImageMatchingDataset # Example dataset
from torch.utils.data import DataLoader
dataset = ImageMatchingDataset(image_paths, ...)
loader = DataLoader(dataset, batch_size=1)
batch_feature_descriptor(loader, device, "disk", "evaluation/disk_descriptors_outputs/")
```

### 2. Feature Matching

After extracting features, you can match them between image pairs using `batch_matcher.py`. This requires a csv containing pair of images, their image paths and we also load the descriptor files which have the descriptors and keypoints and then load it via Dataloader to match it via FLANN or LightGlue.

**Note:** There are several demo files available that have the snippets for running the pipeline for each combination of descriptors and matchers for the whole dataset. For some idiotic reasons, some of the code is only available in the feature branch __evaluation_feature__ , so use that branch for running LightGlue and DISK. But seperate `.ipynb` files are available for each combination for demo run.


### 3. Clustering

Clustering follows feature matching. The `clustering/cluster_images.py` would be used for this step, generating cluster JSON files.

### 4. Evaluation

To evaluate the generated clusters, use the `evaluation/eval.py` script and the snippets in image_matching.ipynb. It takes the cluster JSON file and a labels CSV file as input.


## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the [LICENSE](LICENSE) file.
