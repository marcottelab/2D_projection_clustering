# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 01:03:13 2022

@author: Meghana
"""
from util.input_functions import get_config, read_data
from cluster_image_embeddings import get_image_embedding, cluster_data, get_cluster_wise_indices, plot_tsne, get_image_wise_cluster_labels

import numpy as np

# data, gt_lines, gt_names = read_data('../data/real_dataset/mixture_2D.mrcs',  '../data/real_dataset/clusters/mixture_classification.txt', '   ')

# fname = '../results/real_all/real_node_embedding_slicem_edge_list_top3k_l1_undirected_0.73/APPNPefficientnet_b1/APPNP_reduced_embeddings.npy'
# all_vectors = np.load(fname)
# lbls = get_image_wise_cluster_labels(all_vectors,gt_lines,index_start=1)
# plot_tsne(all_vectors,'test',lbls, gt_names)

data, gt_lines, gt_names = read_data('../data/synthetic_dataset/synthetic_2D.mrcs',  '../data/synthetic_dataset/clusters/synthetic_true_clustering.txt', '\t')

fname = '../results/synthetic_all/synthetic_combined_externally_slicem_cosine_directed_0.84/siamesewys/siamese_reduced_embeddings.npy'
all_vectors = np.load(fname)
lbls = get_image_wise_cluster_labels(all_vectors,gt_lines,index_start=0)
plot_tsne(all_vectors,'test',lbls, gt_names)