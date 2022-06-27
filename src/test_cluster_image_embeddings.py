# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:01:14 2021

@author: Meghana
"""
from util.input_functions import get_config, read_data
from util.evaluation_functions import evaluate_clusters
from cluster_image_embeddings import get_image_embedding, cluster_data, get_cluster_wise_indices, plot_tsne, get_image_wise_cluster_labels
from sklearn.cluster import DBSCAN

import numpy as np


def test_get_config():
    assert get_config(dataset='real') == ('../data/real_dataset/mixture_2D.mrcs', '../data/real_dataset/clusters/mixture_classification.txt', '   ', 1, 'real',', '), get_config(dataset='real')
    assert get_config(dataset='synthetic') == ('../data/synthetic_dataset/synthetic_2D.mrcs', '../data/synthetic_dataset/clusters/synthetic_true_clustering.txt', '\t', 0, 'synthetic',', '),get_config(dataset='synthetic')


def test_read_data():        

    data, gt_lines, gt_names = read_data('../data/real_dataset/mixture_2D.mrcs',  '../data/real_dataset/clusters/mixture_classification.txt', '   ')
    assert len(data) == 100, len(data)
    assert np.array(data[0]).shape == (96,96), np.array(data[0]).shape
    assert gt_names == ['80S', '60S', '40S', 'Apo'], gt_names
    assert len(gt_lines[0]) == 28, len(gt_lines[0])  
    
    data, gt_lines, gt_names = read_data('../data/synthetic_dataset/synthetic_2D.mrcs',  '../data/synthetic_dataset/clusters/synthetic_true_clustering.txt', '\t')
    assert len(data) == 204
    assert np.array(data[0]).shape == (350,350)
    assert gt_names[0] == '1A0I', gt_names[0]
    assert len(gt_lines[0]) == 8 
    
    return data[0:2], data, gt_lines, gt_names


def test_get_image_embedding(test_data):
    vectors = get_image_embedding(test_data,embedding_model = 'resnet-18')
    assert len(vectors) == len(test_data) # 2
    assert len(vectors[0]) == 512
    return vectors


def test_cluster_data(test_vectors):
    clusterwise_indices = get_cluster_wise_indices(1, np.array([-1, -1]))
    assert clusterwise_indices == [[0,1]], clusterwise_indices
    n_clus, clusterwise_indices_str,unsupervised_score_silhouette,unsupervised_score_calinski_harabasz, unsupervised_score_davies_bouldin = cluster_data(test_vectors,DBSCAN(),0)
    assert n_clus == 1, n_clus
    assert clusterwise_indices_str == [({'0', '1'}, 1)]
    print(unsupervised_score_silhouette)
    print(unsupervised_score_calinski_harabasz)
    print(unsupervised_score_davies_bouldin)
 
    
def test_evaluate_clusters():
    eval_metrics = evaluate_clusters(clusterwise_indices_start_str=[({'0', '1'}, 1)],gt_lines=[{'0','1'}],n_clus=1,clustering_method=DBSCAN(),out_dir='./',n_true_clus=1,gt_names='8',main_results_dir='../results',suffix='',plot_hist_flag=1,with_junk=0)
    eval_metrics_true = {'No. of matches in FMM w/o junk': 1, 'FMM Precision w/o junk': 1.0, 'FMM Recall w/o junk': 1.0, 'FMM F1 score w/o junk': 1.0, 'CMFF w/o junk': 1.0, 'Qi Precision w/o junk': 1.0, 'Qi Recall w/o junk': 1.0, 'Qi F1 score w/o junk': 1.0, 'No. of clusters w/o junk': 1}
    assert eval_metrics == eval_metrics_true, eval_metrics
    
    
def test_plot_tsne(test_vectors, gt_lines, gt_names, index_start = 1):
    lbls = get_image_wise_cluster_labels(test_vectors,gt_lines,index_start)
    assert plot_tsne(test_vectors,'test',lbls, gt_names) == None
    

if __name__ == "__main__":    
    test_get_config()
    test_data, all_data, gt_lines, gt_names  = test_read_data()
    # test_vectors= test_get_image_embedding(test_data)
    all_vectors = test_get_image_embedding(all_data)
    # test_cluster_data(test_vectors)
    # test_evaluate_clusters()
    test_plot_tsne(all_vectors, gt_lines, gt_names)
    print('Tests passed')