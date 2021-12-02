# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:01:14 2021

@author: Meghana
"""
from cluster_image_embeddings import get_config, read_data,get_image_embedding, cluster_data, get_cluster_wise_indices, evaluate_clusters
from sklearn.cluster import DBSCAN
import numpy as np


def test_get_config():
    assert get_config(dataset='real') == ('./data/real_dataset/mixture_2D.mrcs', './data/real_dataset/mixture_classification.txt', '   ', 1, 'results_real_dataset')
    assert get_config(dataset='synthetic') == ('./data/synthetic_dataset/synthetic_2D.mrcs', './data/synthetic_dataset/synthetic_true_clustering.txt', '\t', 0, 'results_synthetic_dataset')


def test_read_data():        
    data, gt_lines, gt_names = read_data('./data/real_dataset/mixture_2D.mrcs',  './data/real_dataset/mixture_classification.txt', '   ')
    assert len(data) == 100, len(data)
    assert np.array(data[0]).shape == (96,96), np.array(data[0]).shape
    assert gt_names == ['80S', '60S', '40S', 'Apo'], gt_names
    assert len(gt_lines[0]) == 28, len(gt_lines[0])
    
    data, gt_lines, gt_names = read_data('./data/synthetic_dataset/synthetic_2D.mrcs',  './data/synthetic_dataset/synthetic_true_clustering.txt', '\t')
    assert len(data) == 204
    assert np.array(data[0]).shape == (350,350)
    assert gt_names[0] == '0'
    assert len(gt_lines[0]) == 8
    
    return data[45:47]


def test_get_image_embedding(test_data):
    vectors = get_image_embedding(test_data,embedding_model = 'resnet-18')
    assert len(vectors) == 2
    assert len(vectors[0]) == 512
    return vectors


def test_cluster_data(test_vectors):
    clusterwise_indices = get_cluster_wise_indices(1, np.array([-1, -1]))
    assert clusterwise_indices == [[0,1]], clusterwise_indices
    n_clus, clusterwise_indices_str,unsupervised_score_silhouette = cluster_data(test_vectors,DBSCAN(),0)
    assert n_clus == 1, n_clus
    assert clusterwise_indices_str == [({'0', '1'}, 1)]
 
    
def test_evaluate_clusters():
    eval_metrics = evaluate_clusters(clusterwise_indices_start_str=[({'0', '1'}, 1)],gt_lines=[{'0','1'}],n_clus=1,clustering_method=DBSCAN(),out_dir='./',n_true_clus=1,gt_names='8')
    eval_metrics_true = {'No. of matches in MMR': 1, 'MMR Precision': 1.0, 'MMR Recall': 1.0, 'MMR F1 score': 1.0, 'Net F1 score': 1.0, 'Qi Precision': 1.0, 'Qi Recall': 1.0, 'Qi F1 score': 1.0, 'No. of clusters': 1}
    assert eval_metrics == eval_metrics_true
    

if __name__ == "__main__":    
    test_get_config()
    test_data = test_read_data()
    test_vectors = test_get_image_embedding(test_data)
    test_cluster_data(test_vectors)
    test_evaluate_clusters()
    print('Tests passed')

