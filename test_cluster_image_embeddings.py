# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:01:14 2021

@author: Meghana
"""
from cluster_image_embeddings import get_config, read_data
import numpy as np

def test_get_config():
    assert get_config(dataset='real') == ('./data/real_dataset/mixture_2D.mrcs', './data/real_dataset/mixture_classification.txt', '   ', 1, 'results_real_dataset')
    assert get_config(dataset='synthetic') == ('./data/synthetic_dataset/synthetic_2D.mrcs', './data/synthetic_dataset/synthetic_true_clustering.txt', '\t', 0, 'results_synthetic_dataset')

def test_read_data():
    data, gt_lines, gt_names = read_data('./data/real_dataset/mixture_2D.mrcs',  './data/real_dataset/mixture_classification.txt', '   ')
    assert len(data) == 100, len(data)
    assert np.array(data[0]).shape == (96,96), np.array(data[0]).shape
    assert gt_names == ['80S', '60S', '40S', 'Apo', 'unk'], gt_names
    assert len(gt_lines[0]) == 28, len(gt_lines[0])
    
    data, gt_lines, gt_names = read_data('./data/synthetic_dataset/synthetic_2D.mrcs',  './data/synthetic_dataset/synthetic_true_clustering.txt', '\t')
    assert len(data) == 204
    assert np.array(data[0]).shape == (350,350)
    assert gt_names[0] == '0'
    assert len(gt_lines[0]) == 8
    
test_get_config()
test_read_data()
print('Tests passed')

