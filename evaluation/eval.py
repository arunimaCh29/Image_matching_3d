# copy from branch "evaluation_feature"

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import upsetplot
from upsetplot import UpSet
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics.cluster import contingency_matrix

'''Enhanced evaluation function for image clustering with comprehensive metrics'''

def load_clusters_from_json(json_path: str) -> Dict[str, Dict[int, List[str]]]:
    """
    Load clusters from JSON file format
    
    Args:
        json_path: Path to JSON file containing clusters
        
    Returns:
        Dictionary mapping dataset names to cluster dictionaries
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_clusters_comprehensive(clusters_json_path: str, labels_path: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of image clusters with multiple metrics
    
    Args:
        clusters_json_path: Path to JSON file containing clusters
        labels_path: Path to train_labels.csv
        
    Returns:
        Dictionary containing comprehensive evaluation metrics
    """
    # Load data
    clusters_data = load_clusters_from_json(clusters_json_path)
    labels_df = pd.read_csv(labels_path)
    
    # Create image to scene/dataset mapping
    image_to_scene = {}
    image_to_dataset = {}
    for _, row in labels_df.iterrows():
        image_to_scene[row['image']] = row['scene']
        image_to_dataset[row['image']] = row['dataset']
    
    results = {}
    
    # Evaluate each dataset separately
    for dataset_name, clusters in clusters_data.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        
        # Filter labels for this dataset
        dataset_labels = labels_df[labels_df['dataset'] == dataset_name]
        
        if dataset_labels.empty:
            print(f"No labels found for dataset: {dataset_name}")
            continue
            
        # Convert clusters to the format expected by evaluation functions
        cluster_dict = {int(k): v for k, v in clusters['clusters'].items()}
        
        # Run comprehensive evaluation
        dataset_results = evaluate_dataset_clusters(
            cluster_dict, dataset_labels, image_to_scene, image_to_dataset
        )
        
        results[dataset_name] = dataset_results
    
    # Calculate overall metrics across all datasets
    results['overall'] = calculate_overall_metrics(results)
    
    return results

def evaluate_dataset_clusters(clusters: Dict[int, List[str]], 
                            dataset_labels: pd.DataFrame,
                            image_to_scene: Dict[str, str],
                            image_to_dataset: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate clusters for a single dataset
    """
    # Get all images in clusters
    all_clustered_images = set()
    for cluster_images in clusters.values():
        all_clustered_images.update(cluster_images)
    
    # Filter dataset labels to only include clustered images
    clustered_labels = dataset_labels[dataset_labels['image'].isin(all_clustered_images)]
    
    if clustered_labels.empty:
        return {"error": "No clustered images found in dataset labels"}
    
    # Basic cluster statistics
    cluster_stats = calculate_cluster_statistics(clusters, image_to_scene, image_to_dataset)
    
    # Scene-based evaluation
    scene_metrics = evaluate_scene_clustering(clusters, clustered_labels)
    
    # Dataset-based evaluation
    dataset_metrics = evaluate_dataset_clustering(clusters, clustered_labels)
    
    # Outlier analysis
    outlier_analysis = analyze_outliers(clusters, image_to_scene, image_to_dataset)
    
    # Clustering quality metrics
    clustering_quality = calculate_clustering_quality_metrics(clusters, clustered_labels)
    
    # Cluster-scene relationship analysis
    cluster_scene_analysis = analyze_cluster_scene_relationship(clusters, clustered_labels)
    
    return {
        'cluster_statistics': cluster_stats,
        'scene_metrics': scene_metrics,
        'dataset_metrics': dataset_metrics,
        'outlier_analysis': outlier_analysis,
        'clustering_quality': clustering_quality,
        'cluster_scene_analysis': cluster_scene_analysis
    }

def calculate_cluster_statistics(clusters: Dict[int, List[str]], 
                               image_to_scene: Dict[str, str],
                               image_to_dataset: Dict[str, str]) -> Dict[str, Any]:
    """Calculate basic cluster statistics"""
    stats = {
        'total_clusters': len(clusters),
        'total_images': sum(len(images) for images in clusters.values()),
        'cluster_sizes': [len(images) for images in clusters.values()],
        'avg_cluster_size': np.mean([len(images) for images in clusters.values()]),
        'std_cluster_size': np.std([len(images) for images in clusters.values()]),
        'min_cluster_size': min(len(images) for images in clusters.values()) if clusters else 0,
        'max_cluster_size': max(len(images) for images in clusters.values()) if clusters else 0
    }
    
    # Scene distribution in clusters
    scene_distribution = defaultdict(int)
    dataset_distribution = defaultdict(int)
    
    for cluster_id, images in clusters.items():
        scenes_in_cluster = set()
        datasets_in_cluster = set()
        
        for image in images:
            if image in image_to_scene:
                scenes_in_cluster.add(image_to_scene[image])
            if image in image_to_dataset:
                datasets_in_cluster.add(image_to_dataset[image])
        
        scene_distribution[len(scenes_in_cluster)] += 1
        dataset_distribution[len(datasets_in_cluster)] += 1
    
    stats['scene_distribution'] = dict(scene_distribution)
    stats['dataset_distribution'] = dict(dataset_distribution)
    
    return stats

def evaluate_scene_clustering(clusters: Dict[int, List[str]], 
                            dataset_labels: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate clustering quality based on scene ground truth"""
    
    # Get scene cardinalities
    scene_sizes = dataset_labels.groupby('scene')['image'].count()
    
    # Calculate metrics for each scene-cluster pair
    scene_cluster_scores = {}
    for scene in scene_sizes.index:
        scene_images = set(dataset_labels[dataset_labels['scene'] == scene]['image'])
        scene_cluster_scores[scene] = []
        
        for cluster_id, cluster_images in clusters.items():
            cluster_image_set = set(cluster_images)
            
            # Calculate precision and recall
            common_images = len(scene_images.intersection(cluster_image_set))
            precision = common_images / len(cluster_images) if cluster_images else 0
            recall = common_images / scene_sizes[scene]
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            scene_cluster_scores[scene].append({
                'cluster_id': cluster_id,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'common_images': common_images,
                'cluster_size': len(cluster_images),
                'scene_size': scene_sizes[scene]
            })
    
    # Greedy assignment of scenes to best clusters
    scene_assignments = {}
    for scene in scene_sizes.index:
        # Sort by F1 score first, then precision, then recall
        sorted_scores = sorted(
            scene_cluster_scores[scene],
            key=lambda x: (-x['f1_score'], -x['precision'], -x['recall'])
        )
        scene_assignments[scene] = sorted_scores[0] if sorted_scores else None
    
    # Calculate overall metrics
    valid_assignments = [a for a in scene_assignments.values() if a is not None]
    
    if valid_assignments:
        overall_precision = np.mean([a['precision'] for a in valid_assignments])
        overall_recall = np.mean([a['recall'] for a in valid_assignments])
        overall_f1 = np.mean([a['f1_score'] for a in valid_assignments])
    else:
        overall_precision = overall_recall = overall_f1 = 0.0
    
    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1,
        'per_scene_results': scene_assignments,
        'scene_cluster_scores': scene_cluster_scores
    }

def evaluate_dataset_clustering(clusters: Dict[int, List[str]], 
                              dataset_labels: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate clustering quality based on dataset ground truth"""
    
    # Get dataset cardinalities
    dataset_sizes = dataset_labels.groupby('dataset')['image'].count()
    
    # Calculate metrics for each dataset-cluster pair
    dataset_cluster_scores = {}
    for dataset in dataset_sizes.index:
        dataset_images = set(dataset_labels[dataset_labels['dataset'] == dataset]['image'])
        dataset_cluster_scores[dataset] = []
        
        for cluster_id, cluster_images in clusters.items():
            cluster_image_set = set(cluster_images)
            
            # Calculate precision and recall
            common_images = len(dataset_images.intersection(cluster_image_set))
            precision = common_images / len(cluster_images) if cluster_images else 0
            recall = common_images / dataset_sizes[dataset]
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            dataset_cluster_scores[dataset].append({
                'cluster_id': cluster_id,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'common_images': common_images,
                'cluster_size': len(cluster_images),
                'dataset_size': dataset_sizes[dataset]
            })
    
    # Greedy assignment of datasets to best clusters
    dataset_assignments = {}
    for dataset in dataset_sizes.index:
        sorted_scores = sorted(
            dataset_cluster_scores[dataset],
            key=lambda x: (-x['f1_score'], -x['precision'], -x['recall'])
        )
        dataset_assignments[dataset] = sorted_scores[0] if sorted_scores else None
    
    # Calculate overall metrics
    valid_assignments = [a for a in dataset_assignments.values() if a is not None]
    
    if valid_assignments:
        overall_precision = np.mean([a['precision'] for a in valid_assignments])
        overall_recall = np.mean([a['recall'] for a in valid_assignments])
        overall_f1 = np.mean([a['f1_score'] for a in valid_assignments])
    else:
        overall_precision = overall_recall = overall_f1 = 0.0
    
    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1,
        'per_dataset_results': dataset_assignments,
        'dataset_cluster_scores': dataset_cluster_scores
    }

def analyze_outliers(clusters: Dict[int, List[str]], 
                    image_to_scene: Dict[str, str],
                    image_to_dataset: Dict[str, str]) -> Dict[str, Any]:
    """Analyze outliers in clusters"""
    
    outlier_analysis = {}
    
    for cluster_id, images in clusters.items():
        if not images:
            continue
            
        # Find the most common scene and dataset in this cluster
        scenes = [image_to_scene.get(img, 'unknown') for img in images]
        datasets = [image_to_dataset.get(img, 'unknown') for img in images]
        
        scene_counts = Counter(scenes)
        dataset_counts = Counter(datasets)
        
        most_common_scene = scene_counts.most_common(1)[0][0]
        most_common_dataset = dataset_counts.most_common(1)[0][0]
        
        # Calculate outliers
        scene_outliers = [img for img in images if image_to_scene.get(img, 'unknown') != most_common_scene]
        dataset_outliers = [img for img in images if image_to_dataset.get(img, 'unknown') != most_common_dataset]
        
        outlier_analysis[cluster_id] = {
            'cluster_size': len(images),
            'most_common_scene': most_common_scene,
            'most_common_dataset': most_common_dataset,
            'scene_outliers': scene_outliers,
            'dataset_outliers': dataset_outliers,
            'scene_outlier_percentage': len(scene_outliers) / len(images) * 100,
            'dataset_outlier_percentage': len(dataset_outliers) / len(images) * 100,
            'scene_distribution': dict(scene_counts),
            'dataset_distribution': dict(dataset_counts)
        }
    
    # Calculate overall outlier statistics
    total_images = sum(len(images) for images in clusters.values())
    total_scene_outliers = sum(len(oa['scene_outliers']) for oa in outlier_analysis.values())
    total_dataset_outliers = sum(len(oa['dataset_outliers']) for oa in outlier_analysis.values())
    
    return {
        'per_cluster_outliers': outlier_analysis,
        'overall_scene_outlier_percentage': total_scene_outliers / total_images * 100 if total_images > 0 else 0,
        'overall_dataset_outlier_percentage': total_dataset_outliers / total_images * 100 if total_images > 0 else 0,
        'total_scene_outliers': total_scene_outliers,
        'total_dataset_outliers': total_dataset_outliers,
        'total_images': total_images
    }

def calculate_clustering_quality_metrics(clusters: Dict[int, List[str]], 
                                       dataset_labels: pd.DataFrame) -> Dict[str, Any]:
    """Calculate additional clustering quality metrics"""
    
    # Prepare ground truth labels
    image_to_scene = dict(zip(dataset_labels['image'], dataset_labels['scene']))
    
    # Create cluster labels for each image
    cluster_labels = []
    ground_truth_labels = []
    
    for cluster_id, images in clusters.items():
        for image in images:
            cluster_labels.append(cluster_id)
            ground_truth_labels.append(image_to_scene.get(image, 'unknown'))
    
    if not cluster_labels:
        return {"error": "No cluster labels found"}
    
    # Convert to numeric labels for sklearn metrics
    unique_clusters = list(set(cluster_labels))
    unique_scenes = list(set(ground_truth_labels))
    
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    scene_to_idx = {s: i for i, s in enumerate(unique_scenes)}
    
    cluster_labels_numeric = [cluster_to_idx[c] for c in cluster_labels]
    ground_truth_numeric = [scene_to_idx[s] for s in ground_truth_labels]
    
    # Calculate metrics
    try:
        ari = adjusted_rand_score(ground_truth_numeric, cluster_labels_numeric)
        nmi = normalized_mutual_info_score(ground_truth_numeric, cluster_labels_numeric)
        homogeneity = homogeneity_score(ground_truth_numeric, cluster_labels_numeric)
        completeness = completeness_score(ground_truth_numeric, cluster_labels_numeric)
        v_measure = v_measure_score(ground_truth_numeric, cluster_labels_numeric)
    except Exception as e:
        return {"error": f"Error calculating clustering metrics: {str(e)}"}
    
    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure
    }

def analyze_cluster_scene_relationship(clusters: Dict[int, List[str]], 
                                     dataset_labels: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the relationship between clusters and scenes
    """
    # Get unique scenes
    unique_scenes = dataset_labels['scene'].unique()
    num_scenes = len(unique_scenes)
    num_clusters = len(clusters)
    
    # Check if number of clusters equals number of scenes
    clusters_equal_scenes = num_clusters == num_scenes
    
    # Create image to scene+dataset mapping (since no multi-scene images)
    image_to_scene_dataset = {}
    for _, row in dataset_labels.iterrows():
        image_to_scene_dataset[row['image']] = (row['scene'], row['dataset'])
    
    # Analyze cluster composition
    cluster_analysis = {}
    for cluster_id, cluster_images in clusters.items():
        # Get scene+dataset combinations for images in this cluster
        cluster_scene_datasets = [image_to_scene_dataset.get(img, ('unknown', 'unknown')) for img in cluster_images]
        
        # Count unique scene+dataset combinations
        unique_scene_datasets = set(cluster_scene_datasets)
        scene_dataset_counts = Counter(cluster_scene_datasets)
        
        # Find the most common scene+dataset combination
        most_common = scene_dataset_counts.most_common(1)[0] if scene_dataset_counts else (None, 0)
        
        cluster_analysis[cluster_id] = {
            'cluster_size': len(cluster_images),
            'unique_scene_datasets': len(unique_scene_datasets),
            'scene_dataset_combinations': list(unique_scene_datasets),
            'most_common_scene_dataset': most_common[0] if most_common[0] else None,
            'most_common_count': most_common[1],
            'purity': most_common[1] / len(cluster_images) if cluster_images else 0,
            'is_pure': len(unique_scene_datasets) == 1
        }
    
    # Calculate overall statistics
    total_images = sum(len(images) for images in clusters.values())
    pure_clusters = sum(1 for analysis in cluster_analysis.values() if analysis['is_pure'])
    mixed_clusters = sum(1 for analysis in cluster_analysis.values() if not analysis['is_pure'])
    
    # Check if each scene has a corresponding cluster
    scene_to_cluster_mapping = {}
    for scene in unique_scenes:
        scene_images = set(dataset_labels[dataset_labels['scene'] == scene]['image'])
        best_cluster = None
        best_overlap = 0
        
        for cluster_id, cluster_images in clusters.items():
            cluster_set = set(cluster_images)
            overlap = len(scene_images.intersection(cluster_set))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = cluster_id
        
        scene_to_cluster_mapping[scene] = {
            'best_cluster': best_cluster,
            'overlap_count': best_overlap,
            'scene_size': len(scene_images),
            'overlap_percentage': best_overlap / len(scene_images) * 100 if scene_images else 0
        }
    
    # Check for one-to-one mapping between scenes and clusters
    cluster_to_scene_mapping = {}
    for cluster_id, analysis in cluster_analysis.items():
        if analysis['is_pure'] and analysis['most_common_scene_dataset']:
            scene = analysis['most_common_scene_dataset'][0]
            cluster_to_scene_mapping[cluster_id] = scene
    
    # Check if we have a perfect one-to-one mapping
    perfect_mapping = (len(cluster_to_scene_mapping) == num_clusters == num_scenes and
                      set(cluster_to_scene_mapping.values()) == set(unique_scenes))
    
    # Create overlap matrix for plotting
    overlap_matrix = create_overlap_matrix(clusters, unique_scenes, dataset_labels)
    
    return {
        'num_scenes': num_scenes,
        'num_clusters': num_clusters,
        'clusters_equal_scenes': clusters_equal_scenes,
        'perfect_scene_cluster_mapping': perfect_mapping,
        'total_images': total_images,
        'pure_clusters': pure_clusters,
        'mixed_clusters': mixed_clusters,
        'cluster_analysis': cluster_analysis,
        'scene_to_cluster_mapping': scene_to_cluster_mapping,
        'cluster_to_scene_mapping': cluster_to_scene_mapping,
        'overlap_matrix': overlap_matrix,  # NEW: Add overlap matrix for plotting
        'primary_key_analysis': {
            'images_with_multiple_scene_datasets': 0,  # No multi-scene images
            'total_unique_images': len(image_to_scene_dataset),
            'multi_scene_percentage': 0.0  # No multi-scene images
        }
    }

def create_overlap_matrix(clusters: Dict[int, List[str]], 
                         unique_scenes: np.ndarray, 
                         dataset_labels: pd.DataFrame) -> np.ndarray:
    """
    Create overlap matrix between clusters and scenes for plotting
    """
    num_clusters = len(clusters)
    num_scenes = len(unique_scenes)
    
    # Create cluster ID to index mapping
    cluster_ids = sorted(clusters.keys())
    cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}
    
    # Create scene to index mapping
    scene_to_idx = {scene: idx for idx, scene in enumerate(unique_scenes)}
    
    # Initialize overlap matrix
    overlap_matrix = np.zeros((num_clusters, num_scenes))
    
    # Fill the matrix
    for cluster_id, cluster_images in clusters.items():
        cluster_idx = cluster_to_idx[cluster_id]
        cluster_set = set(cluster_images)
        
        for scene in unique_scenes:
            scene_images = set(dataset_labels[dataset_labels['scene'] == scene]['image'])
            overlap = len(scene_images.intersection(cluster_set))
            scene_idx = scene_to_idx[scene]
            overlap_matrix[cluster_idx, scene_idx] = overlap
    
    return overlap_matrix


def calculate_overall_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall metrics across all datasets"""
    
    # Collect metrics from all datasets
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_aris = []
    all_nmis = []
    all_homogeneities = []
    all_completenesses = []
    all_v_measures = []
    
    for dataset_name, dataset_results in results.items():
        if dataset_name == 'overall' or 'error' in dataset_results:
            continue
            
        # Scene metrics
        if 'scene_metrics' in dataset_results:
            scene_metrics = dataset_results['scene_metrics']
            all_precisions.append(scene_metrics.get('overall_precision', 0))
            all_recalls.append(scene_metrics.get('overall_recall', 0))
            all_f1_scores.append(scene_metrics.get('overall_f1_score', 0))
        
        # Clustering quality metrics
        if 'clustering_quality' in dataset_results and 'error' not in dataset_results['clustering_quality']:
            quality_metrics = dataset_results['clustering_quality']
            all_aris.append(quality_metrics.get('adjusted_rand_index', 0))
            all_nmis.append(quality_metrics.get('normalized_mutual_info', 0))
            all_homogeneities.append(quality_metrics.get('homogeneity', 0))
            all_completenesses.append(quality_metrics.get('completeness', 0))
            all_v_measures.append(quality_metrics.get('v_measure', 0))
    
    # Calculate averages
    overall_metrics = {
        'average_precision': np.mean(all_precisions) if all_precisions else 0,
        'average_recall': np.mean(all_recalls) if all_recalls else 0,
        'average_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0,
        'average_ari': np.mean(all_aris) if all_aris else 0,
        'average_nmi': np.mean(all_nmis) if all_nmis else 0,
        'average_homogeneity': np.mean(all_homogeneities) if all_homogeneities else 0,
        'average_completeness': np.mean(all_completenesses) if all_completenesses else 0,
        'average_v_measure': np.mean(all_v_measures) if all_v_measures else 0,
        'num_datasets_evaluated': len([k for k in results.keys() if k != 'overall' and 'error' not in results[k]])
    }
    
    return overall_metrics

def evaluate_clusters(clusters: Dict[int, List[str]], labels_path: str) -> Dict:
    """
    Legacy function for backward compatibility
    """
    # This is the original function - keeping it for compatibility
    labels_df = pd.read_csv(labels_path)
    
    # Get scene cardinalities
    scene_sizes = labels_df.groupby('scene')['image'].count()
    
    # Calculate mAA and clustering scores for each scene-cluster pair
    scene_cluster_scores = {}
    for scene in scene_sizes.index:
        scene_cluster_scores[scene] = []
        scene_images = set(labels_df[labels_df['scene'] == scene]['image'])
        
        for cluster_id, cluster_images in clusters.items():
            cluster_image_set = set(cluster_images)
            
            # Calculate mAA (recall)
            common_images = len(scene_images.intersection(cluster_image_set))
            mAA = common_images / scene_sizes[scene]
            
            # Calculate clustering score (precision)
            clustering_score = common_images / len(cluster_images) if cluster_images else 0
            
            scene_cluster_scores[scene].append({
                'cluster_id': cluster_id,
                'mAA': mAA,
                'clustering_score': clustering_score
            })

    # Greedy assignment of scenes to best clusters
    scene_assignments = {}
    for scene in scene_sizes.index:
        # Sort by mAA first, then clustering score
        sorted_scores = sorted(
            scene_cluster_scores[scene],
            key=lambda x: (-x['mAA'], -x['clustering_score'])
        )
        scene_assignments[scene] = sorted_scores[0]
    
    # Calculate overall scores
    total_scenes = len(scene_sizes)
    overall_mAA = sum(assignment['mAA'] for assignment in scene_assignments.values()) / total_scenes
    overall_clustering = sum(assignment['clustering_score'] for assignment in scene_assignments.values()) / total_scenes
    
    # Calculate combined score (harmonic mean)
    combined_score = 2 * (overall_mAA * overall_clustering) / (overall_mAA + overall_clustering) if (overall_mAA + overall_clustering) > 0 else 0
    
    results = {
        'overall_mAA': overall_mAA,
        'overall_clustering_score': overall_clustering,
        'combined_score': combined_score,
        'per_scene_results': scene_assignments
    }
    
    return results

def check_cluster_scene_relationship(clusters_json_path: str, labels_path: str) -> Dict[str, Any]:
    """
    Check if number of clusters equals number of scenes and verify primary key structure
    
    Args:
        clusters_json_path: Path to JSON file containing clusters
        labels_path: Path to train_labels.csv
        
    Returns:
        Dictionary containing analysis results
    """
    # Load data
    clusters_data = load_clusters_from_json(clusters_json_path)
    labels_df = pd.read_csv(labels_path)
    
    results = {}
    
    for dataset_name, clusters in clusters_data.items():
        print(f"\nAnalyzing dataset: {dataset_name}")
        
        # Filter labels for this dataset
        dataset_labels = labels_df[labels_df['dataset'] == dataset_name]
        
        if dataset_labels.empty:
            print(f"No labels found for dataset: {dataset_name}")
            continue
        
        # Get unique scenes in this dataset
        unique_scenes = dataset_labels['scene'].unique()
        num_scenes = len(unique_scenes)
        num_clusters = len(clusters)
        
        # Check if number of clusters equals number of scenes
        clusters_equal_scenes = num_clusters == num_scenes
        
        # Since there are no multi-scene images, each image belongs to exactly one scene+dataset
        # Create image to scene+dataset mapping
        image_to_scene_dataset = {}
        for _, row in dataset_labels.iterrows():
            image_to_scene_dataset[row['image']] = (row['scene'], row['dataset'])
        
        # Analyze cluster composition
        cluster_analysis = {}
        for cluster_id, cluster_images in clusters.items():
            # Get scene+dataset combinations for images in this cluster
            cluster_scene_datasets = [image_to_scene_dataset.get(img, ('unknown', 'unknown')) for img in cluster_images]
            
            # Count unique scene+dataset combinations
            unique_scene_datasets = set(cluster_scene_datasets)
            scene_dataset_counts = Counter(cluster_scene_datasets)
            
            # Find the most common scene+dataset combination
            most_common = scene_dataset_counts.most_common(1)[0] if scene_dataset_counts else (None, 0)
            
            cluster_analysis[cluster_id] = {
                'cluster_size': len(cluster_images),
                'unique_scene_datasets': len(unique_scene_datasets),
                'scene_dataset_combinations': list(unique_scene_datasets),
                'most_common_scene_dataset': most_common[0] if most_common[0] else None,
                'most_common_count': most_common[1],
                'purity': most_common[1] / len(cluster_images) if cluster_images else 0,
                'is_pure': len(unique_scene_datasets) == 1
            }
        
        # Calculate overall statistics
        total_images = sum(len(images) for images in clusters.values())
        pure_clusters = sum(1 for analysis in cluster_analysis.values() if analysis['is_pure'])
        mixed_clusters = sum(1 for analysis in cluster_analysis.values() if not analysis['is_pure'])
        
        # Check if each scene has a corresponding cluster
        scene_to_cluster_mapping = {}
        for scene in unique_scenes:
            scene_images = set(dataset_labels[dataset_labels['scene'] == scene]['image'])
            best_cluster = None
            best_overlap = 0
            
            for cluster_id, cluster_images in clusters.items():
                cluster_set = set(cluster_images)
                overlap = len(scene_images.intersection(cluster_set))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = cluster_id
            
            scene_to_cluster_mapping[scene] = {
                'best_cluster': best_cluster,
                'overlap_count': best_overlap,
                'scene_size': len(scene_images),
                'overlap_percentage': best_overlap / len(scene_images) * 100 if scene_images else 0
            }
        
        # Check for one-to-one mapping between scenes and clusters
        cluster_to_scene_mapping = {}
        for cluster_id, analysis in cluster_analysis.items():
            if analysis['is_pure'] and analysis['most_common_scene_dataset']:
                scene = analysis['most_common_scene_dataset'][0]
                cluster_to_scene_mapping[cluster_id] = scene
        
        # Check if we have a perfect one-to-one mapping
        perfect_mapping = (len(cluster_to_scene_mapping) == num_clusters == num_scenes and
                          set(cluster_to_scene_mapping.values()) == set(unique_scenes))
        
        results[dataset_name] = {
            'num_scenes': num_scenes,
            'num_clusters': num_clusters,
            'clusters_equal_scenes': clusters_equal_scenes,
            'perfect_scene_cluster_mapping': perfect_mapping,
            'total_images': total_images,
            'pure_clusters': pure_clusters,
            'mixed_clusters': mixed_clusters,
            'cluster_analysis': cluster_analysis,
            'scene_to_cluster_mapping': scene_to_cluster_mapping,
            'cluster_to_scene_mapping': cluster_to_scene_mapping,
            'primary_key_analysis': {
                'images_with_multiple_scene_datasets': 0,  # No multi-scene images
                'total_unique_images': len(image_to_scene_dataset),
                'multi_scene_percentage': 0.0  # No multi-scene images
            }
        }
    
    return results

def print_cluster_scene_analysis(results: Dict[str, Any]) -> None:
    """
    Print detailed analysis of cluster-scene relationship
    """
    print("\n" + "="*80)
    print("CLUSTER-SCENE RELATIONSHIP ANALYSIS")
    print("="*80)
    
    for dataset_name, analysis in results.items():
        print(f"\n{'-'*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'-'*60}")
        
        print(f"Number of Scenes: {analysis['num_scenes']}")
        print(f"Number of Clusters: {analysis['num_clusters']}")
        print(f"Clusters = Scenes: {analysis['clusters_equal_scenes']}")
        print(f"Perfect Scene-Cluster Mapping: {analysis['perfect_scene_cluster_mapping']}")
        print(f"Total Images: {analysis['total_images']}")
        print(f"Pure Clusters (single scene+dataset): {analysis['pure_clusters']}")
        print(f"Mixed Clusters (multiple scene+dataset): {analysis['mixed_clusters']}")
        
        # Primary key analysis
        pk_analysis = analysis['primary_key_analysis']
        print(f"\nPrimary Key Analysis (scene + dataset):")
        print(f"  Images with multiple scene+dataset combinations: {pk_analysis['images_with_multiple_scene_datasets']}")
        print(f"  Total unique images: {pk_analysis['total_unique_images']}")
        print(f"  Multi-scene percentage: {pk_analysis['multi_scene_percentage']:.2f}%")
        
        # Scene to cluster mapping
        print(f"\nScene to Cluster Mapping:")
        for scene, mapping in analysis['scene_to_cluster_mapping'].items():
            print(f"  Scene '{scene}':")
            print(f"    Best cluster: {mapping['best_cluster']}")
            print(f"    Overlap: {mapping['overlap_count']}/{mapping['scene_size']} ({mapping['overlap_percentage']:.1f}%)")
        
        # Cluster to scene mapping
        print(f"\nCluster to Scene Mapping:")
        for cluster_id, scene in analysis['cluster_to_scene_mapping'].items():
            print(f"  Cluster {cluster_id} -> Scene '{scene}'")
        
        # Cluster analysis
        print(f"\nCluster Analysis:")
        for cluster_id, cluster_info in analysis['cluster_analysis'].items():
            print(f"  Cluster {cluster_id}:")
            print(f"    Size: {cluster_info['cluster_size']}")
            print(f"    Unique scene+dataset combinations: {cluster_info['unique_scene_datasets']}")
            print(f"    Most common: {cluster_info['most_common_scene_dataset']} ({cluster_info['most_common_count']} images)")
            print(f"    Purity: {cluster_info['purity']:.2f}")
            print(f"    Is Pure: {cluster_info['is_pure']}")

def run_comprehensive_evaluation(clusters_json_path: str, labels_path: str) -> None:
    """
    Run comprehensive evaluation and print results
    """
    print("Running comprehensive cluster evaluation...")
    results = evaluate_clusters_comprehensive(clusters_json_path, labels_path)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CLUSTER EVALUATION RESULTS")
    print("="*80)
    
    # Print overall results
    if 'overall' in results:
        overall = results['overall']
        print(f"\nOVERALL METRICS ACROSS ALL DATASETS:")
        print(f"  Average Precision: {overall['average_precision']:.4f}")
        print(f"  Average Recall: {overall['average_recall']:.4f}")
        print(f"  Average F1 Score: {overall['average_f1_score']:.4f}")
        print(f"  Average ARI: {overall['average_ari']:.4f}")
        print(f"  Average NMI: {overall['average_nmi']:.4f}")
        print(f"  Average Homogeneity: {overall['average_homogeneity']:.4f}")
        print(f"  Average Completeness: {overall['average_completeness']:.4f}")
        print(f"  Average V-Measure: {overall['average_v_measure']:.4f}")
        print(f"  Datasets Evaluated: {overall['num_datasets_evaluated']}")
    
    # Print per-dataset results and create plots
    for dataset_name, dataset_results in results.items():
        if dataset_name == 'overall' or 'error' in dataset_results:
            continue
            
        print(f"\n{'-'*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'-'*60}")
        
        # Cluster statistics
        if 'cluster_statistics' in dataset_results:
            stats = dataset_results['cluster_statistics']
            print(f"  Total Clusters: {stats['total_clusters']}")
            print(f"  Total Images: {stats['total_images']}")
            print(f"  Average Cluster Size: {stats['avg_cluster_size']:.2f}")
            print(f"  Cluster Size Std: {stats['std_cluster_size']:.2f}")
            print(f"  Min/Max Cluster Size: {stats['min_cluster_size']}/{stats['max_cluster_size']}")
        
        # Cluster-scene relationship analysis
        if 'cluster_scene_analysis' in dataset_results:
            csa = dataset_results['cluster_scene_analysis']
            print(f"\n  CLUSTER-SCENE RELATIONSHIP:")
            print(f"    Number of Scenes: {csa['num_scenes']}")
            print(f"    Number of Clusters: {csa['num_clusters']}")
            print(f"    Clusters = Scenes: {csa['clusters_equal_scenes']}")
            print(f"    Perfect Scene-Cluster Mapping: {csa['perfect_scene_cluster_mapping']}")
            print(f"    Pure Clusters: {csa['pure_clusters']}")
            print(f"    Mixed Clusters: {csa['mixed_clusters']}")
            
            # Show scene-to-cluster mapping
            print(f"    Scene-to-Cluster Mapping:")
            for scene, mapping in csa['scene_to_cluster_mapping'].items():
                print(f"      Scene '{scene}' -> Cluster {mapping['best_cluster']} ({mapping['overlap_percentage']:.1f}% overlap)")
            
            # Show cluster purity details
            print(f"    Cluster Purity Details:")
            for cluster_id, analysis in csa['cluster_analysis'].items():
                print(f"      Cluster {cluster_id}: {analysis['cluster_size']} images, "
                      f"Purity: {analysis['purity']:.2f}, "
                      f"Pure: {analysis['is_pure']}, "
                      f"Scene: {analysis['most_common_scene_dataset'][0] if analysis['most_common_scene_dataset'] else 'Unknown'}")
            
            # Create plots
            '''print(f"\n  Generating interactive plots for {dataset_name}...")
            
            # Plot 1: UpSet Plot for cluster-scene overlap
            if 'overlap_matrix' in csa:
                cluster_ids = sorted(csa['cluster_analysis'].keys())
                scene_names = list(csa['scene_to_cluster_mapping'].keys())
                plot_cluster_scene_overlap(csa['overlap_matrix'], cluster_ids, scene_names, dataset_name)
            
            # Plot 2: Interactive Plotly heatmap
            plot_cluster_scene_plotly(csa['overlap_matrix'], cluster_ids, scene_names, dataset_name)
            
            # Plot 3: Interactive network plot
            plot_cluster_scene_network_plotly(csa['cluster_analysis'], csa['scene_to_cluster_mapping'], dataset_name)
            
            # Plot 4: Interactive cluster purity analysis
            plot_cluster_purity_plotly(csa['cluster_analysis'], dataset_name)
            
            # Plot 5: Interactive scene coverage
            plot_scene_coverage_plotly(csa['scene_to_cluster_mapping'], dataset_name)'''
        
        # Scene metrics
        if 'scene_metrics' in dataset_results:
            scene_metrics = dataset_results['scene_metrics']
            print(f"  Scene-based Precision: {scene_metrics['overall_precision']:.4f}")
            print(f"  Scene-based Recall: {scene_metrics['overall_recall']:.4f}")
            print(f"  Scene-based F1 Score: {scene_metrics['overall_f1_score']:.4f}")
        
        # Dataset metrics
        if 'dataset_metrics' in dataset_results:
            dataset_metrics = dataset_results['dataset_metrics']
            print(f"  Dataset-based Precision: {dataset_metrics['overall_precision']:.4f}")
            print(f"  Dataset-based Recall: {dataset_metrics['overall_recall']:.4f}")
            print(f"  Dataset-based F1 Score: {dataset_metrics['overall_f1_score']:.4f}")
        
        # Outlier analysis
        if 'outlier_analysis' in dataset_results:
            outlier_analysis = dataset_results['outlier_analysis']
            print(f"  Scene Outlier Percentage: {outlier_analysis['overall_scene_outlier_percentage']:.2f}%")
            print(f"  Dataset Outlier Percentage: {outlier_analysis['overall_dataset_outlier_percentage']:.2f}%")
        
        # Clustering quality
        if 'clustering_quality' in dataset_results and 'error' not in dataset_results['clustering_quality']:
            quality = dataset_results['clustering_quality']
            print(f"  Adjusted Rand Index: {quality['adjusted_rand_index']:.4f}")
            print(f"  Normalized Mutual Info: {quality['normalized_mutual_info']:.4f}")
            print(f"  Homogeneity: {quality['homogeneity']:.4f}")
            print(f"  Completeness: {quality['completeness']:.4f}")
            print(f"  V-Measure: {quality['v_measure']:.4f}")
        