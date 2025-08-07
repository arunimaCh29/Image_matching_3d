import networkx as nx
import numpy as np

def build_graph(images_name, matching_data, threshold=None, thresConst=1.5):
    '''
        Build graph using NetworkX for clustering

        Args:
            images_name (list)     : list of image name in a dataset
            matching_data (list)   : list of data of image matching
            threshold (None / int) : threshold for the graph edge
            thresConst (number)    : a constant to be multiplied for alpha, used if threshold is not None
        
        Return:
            G : graph of the images in a dataset with edges based on number of matches
    '''

    G = nx.Graph()

    for idx, name in enumerate(images_name):
        G.add_node(name)
    
    num_good_matches = [len(data["good_matches"]) for data in matching_data]

    if threshold is not None:
        alpha = threshold
    else:
        alpha = thresConst * np.median(num_good_matches)
    # alpha = threshold if threshold not None else (thresConst * np.median(num_good_matches))

    for result in matching_data:
        image0 = result["images_name"][0]
        image1 = result["images_name"][1]
        count_good_matches = len(result["good_matches"])
        if count_good_matches >= alpha:
            G.add_edge(image0, image1, weight=count_good_matches)
    
    return G

def graph_clustering(G, threshold=None):
    '''
        Cluster graph with community detection using Louvain method

        Args:
            G (graph)   : networkx graph
            threshold   : number of minimum image to filter the clusters
    '''

    communities = nx.community.louvain_communities(G) # return of louvain_communities is a list of dict

    if threshold is not None:
        alpha = threshold
    else:
        len_communities = [len(c) for c in communities]
        alpha = np.median(len_communities)
        # if the median below 3 set alpha as 3
        if alpha < 3:
            alpha = 3
    
    clusters = {}
    outliers = {}

    for idx, com in enumerate(communities):
        if len(com) >= alpha:
            clusters[idx] = list(com)
            # clusters.append({idx : com})
        else:
            outliers[idx] = list(com)
            # outliers.append({idx : com})
    
    return clusters, outliers