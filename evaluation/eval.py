1import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict

'''Maybe using image indexes will be better than image names and the evaluation is developed according to our interpretation of evaluation description of Image Matching competition'''

def evaluate_clusters(clusters: Dict[int, List[str]], labels_path: str) -> Dict:
    """
    Evaluate image clusters using the competition metric
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of image names
        labels_path: Path to train_labels.csv
        
    Returns:
        Dictionary containing evaluation metrics including mAA, clustering score,
        and combined score
    """
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
            print(cluster_image_set)
            
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

