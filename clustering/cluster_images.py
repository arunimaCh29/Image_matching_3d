import networkx as nx
import numpy as np
from collections import defaultdict
import os
import community as community_louvain
from pyvis.network import Network
import ast

def build_graph(matches_df, labels_df, matcher):
    dataset_name = matches_df["dataset"].unique()
    
    graphs = {}
    for dataset in dataset_name:
        dataset_df = matches_df[matches_df["dataset"] == dataset]
        images_name = list(labels_df[labels_df["dataset"] == dataset]["image"])
    
        G = nx.Graph()
        for idx, name in enumerate(images_name):
            G.add_node(name)
        graphs[dataset] = G
    
        for idx, row in matches_df[matches_df["dataset"] == dataset].iterrows():
            image1 = row["image1"]
            image2 = row["image2"]
            if matcher == "flann":
                matches = np.array(ast.literal_eval(row["matches_idx"]))
            elif matcher == "lightglue":
                matches = np.array(ast.literal_eval(row["matches"]))
                                   
            if row["filtered_points0"] is None and row["filtered_points1"] is None and row["ransac_mask"] is None:
                continue
            
            count_filtered_match = len(matches[row["ransac_mask"]])
            G.add_edge(image1, image2, weight=count_filtered_match)

    return graphs

# def build_graph(images_name, matching_data, matcher, threshold=None, thresConst=1.5):
#     '''
#         Build graph using NetworkX for clustering

#         Args:
#             images_name (list)     : list of image name in a dataset
#             matching_data (list)   : list of data of image matching
#             threshold (None / int) : threshold for the graph edge
#             thresConst (number)    : a constant to be multiplied for alpha, used if threshold is not None
        
#         Return:
#             G : graph of the images in a dataset with edges based on number of matches
#     '''

#     G = nx.Graph()

#     for idx, name in enumerate(images_name):
#         G.add_node(name)

#     # if matcher == "flann":
#     #     num_good_matches = [len(data["matches_idx"]) for data in matching_data]
#     # elif matcher == "lightglue":
#     #     num_good_matches = [len(data["matches"]) for data in matching_data]
#     num_good_matches = [len(data["good_matches"]) for data in matching_data]

#     if threshold is not None:
#         alpha = threshold
#     else:
#         alpha = thresConst * np.median(num_good_matches)
#     # alpha = threshold if threshold not None else (thresConst * np.median(num_good_matches))

#     for result in matching_data:
#         image0 = result["image1_name"][0]
#         image1 = result["image2_name"][0]
#         count_good_matches = len(result["good_matches"])
#         if count_good_matches >= alpha:
#             G.add_edge(image0, image1, weight=count_good_matches)
    
#     return G

def graph_clustering(graphs, threshold=None):
    clustering = {}
    for graph in graphs:
        gr = graphs[graph]
        communities = community_louvain.best_partition(gr) # output: dict with image name as the key and the community index as the value, e.g.: {imageA : 0, imageB : 0, imageC : 1}
        # invert the communities output so that it is a dict with the community index as the key and list of image name as the value, e.g. {0 : [imageA, imageB], 1 : [imageC]}
        inverted_communities = defaultdict(list)
        for k, v in communities.items():
            inverted_communities[v].append(k)
        
        len_communities = [len(inverted_communities[c]) for c in inverted_communities]

        if threshold is not None:
            alpha = threshold
        else:
            alpha = np.median(len_communities)
            # if the median below 3 set alpha as 3
            if alpha < 3:
                alpha = 3
        
        clusters = {}
        outliers = {}
        
        for com_idx in inverted_communities:
            com = inverted_communities[com_idx]
            if len(com) >= alpha:
                clusters[com_idx] = com
            else:
                outliers[com_idx] = com
        
        clustering[graph] = {
            "clusters" : clusters,
            "outliers" : outliers,
            "communities" : communities,
            "inverted_communities" : inverted_communities,
        }

    return clustering

# def graph_clustering(G, threshold=None):
#     '''
#         Cluster graph with community detection using Louvain method

#         Args:
#             G (graph)   : networkx graph
#             threshold   : number of minimum image to filter the clusters
#     '''

#     communities = nx.community.louvain_communities(G) # return of louvain_communities is a list of dict

#     if threshold is not None:
#         alpha = threshold
#     else:
#         len_communities = [len(c) for c in communities]
#         alpha = np.median(len_communities)
#         # if the median below 3 set alpha as 3
#         if alpha < 3:
#             alpha = 3
    
#     clusters = {}
#     outliers = {}

#     for idx, com in enumerate(communities):
#         if len(com) >= alpha:
#             clusters[idx] = list(com)
#             # clusters.append({idx : com})
#         else:
#             outliers[idx] = list(com)
#             # outliers.append({idx : com})
    
#     return clusters, outliers

def interactive_graph(graphs, clustering, output_dir):
    for data in clustering:
        graph = graphs[data]
        cluster = clustering[data]
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
        communities = cluster["communities"]
        # Add nodes with community colors
        for node, comm in communities.items():
            net.add_node(
                node,
                label=str(comm), # or str(node) if want the image name as the label
                color=f"hsl({comm * 50 % 360}, 70%, 50%)",  # different color per community
                title=str(node) # hover title
            )
        
        # Add edges
        for u, v in graph.edges():
            net.add_edge(u, v)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"visualization_cluster_{data}.html")
        
        net.show(out_path)