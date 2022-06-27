# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:35:03 2022

@author: Meghana
"""
from loguru import logger

import mrcfile
import numpy as np


def get_config(dataset='real'):
    '''
    Get input and output config parameters for real or synthetic datasets
    
    Parameters:
    dataset (string): can be 'real' or 'synthetic'
    
    Returns:
    images_file_name (string): input mrcs file containing stack of images
    images_true_labels (string): true cluster labels in the format: name list_of_image_indices
    sep (string): Separater between name of cluster and its members
    index_start (integer): Index at which images are started being numbered, ex: 0, 1, etc
    out_dir (string): output directory name for results (will be created if it does not exist)

    '''
    if dataset == 'real':
        images_file_name = '../data/real_dataset/mixture_2D.mrcs' 
        images_true_labels = '../data/real_dataset/clusters/mixture_classification.txt'
        sep ='   '
        sep2=', '
        index_start = 1
        out_dir = 'real'
    elif dataset == 'synthetic_more_projs':
        images_file_name = '../data/synthetic_more_projs_dataset/synthetic_more_projections.mrcs' 
        images_true_labels = '../data/synthetic_more_projs_dataset/clusters/true_clustering.txt'
        sep = '\t'
        sep2=','
        index_start = 0
        out_dir = 'synthetic_more_projs'      
    elif dataset == 'synthetic_more_projs_wo_4v6c':
        images_file_name = '../data/synthetic_more_projs_wo_4v6c/synthetic_more_projections_wo_4v6c.mrcs' 
        images_true_labels = '../data/synthetic_more_projs_wo_4v6c/clusters/true_clustering.txt'
        sep = '\t'
        sep2=','
        index_start = 0
        out_dir = 'synthetic_more_projs_wo_4v6c'      
    elif dataset == 'synthetic_more_projs_noisy':
        images_file_name = '../data/synthetic_more_projs_noisy_dataset/synthetic_noisy.mrcs' 
        images_true_labels = '../data/synthetic_more_projs_noisy_dataset/clusters/true_clustering.txt'
        sep = '\t'
        sep2=','
        index_start = 0
        out_dir = 'synthetic_more_projs_noisy'      
    elif dataset == 'synthetic_noisy':
        images_file_name = '../data/synthetic_noisy_dataset/synthetic_noisy.mrcs' 
        images_true_labels = '../data/synthetic_noisy_dataset/clusters/synthetic_true_clustering.txt'
        sep = '\t'
        sep2=', '
        index_start = 0
        out_dir = 'synthetic_noisy'          
    else: # synthetic
        images_file_name = '../data/synthetic_dataset/synthetic_2D.mrcs' 
        images_true_labels = '../data/synthetic_dataset/clusters/synthetic_true_clustering.txt'
        sep = '\t'
        sep2=', '
        index_start = 0
        out_dir = 'synthetic' 

        
    return images_file_name,images_true_labels,sep,index_start,out_dir,sep2


def read_clusters(images_true_labels,sep,sep2=', '):
    '''
    Reads clustered labels

    Parameters:    
    images_true_labels (string): true cluster labels in the format: name list_of_image_indices
    sep (string): Separater between name of cluster and its members

    Returns:
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order
    
    '''    
    with open(images_true_labels) as f:
        raw_lines = f.readlines()
        list_of_tuples = [line.rstrip().split(sep) for line in raw_lines]
        gt_names = [entry[0] for entry in list_of_tuples]
        #gt_names = [line[0:sep] for line in raw_lines]
        #gt_lines = [set(line.rstrip()[sep:-1].split(', ')) for line in raw_lines]
        # Assuming format is [4,5,23,..]
        gt_lines = [set(line[1].lstrip()[1:-1].split(sep2)) for line in list_of_tuples]
        
    # Remove unknown complex for evaluation
    for i, name in enumerate(gt_names):
        if name == 'unk':
            gt_names.pop(i)
            gt_lines.pop(i)
            
    return gt_lines, gt_names


def read_data(images_file_name, images_true_labels, sep, sep2=', '):
    '''
    Reads MRC data and ground truth labels

    Parameters:    
    images_file_name (string): input mrcs file containing stack of images
    images_true_labels (string): filename of file with true cluster labels in the format: name list_of_image_indices
    sep (string): Separater between name of cluster and its members

    Returns:
    data (numpy.ndarray): Image stack of 2D numpy arrays
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order
    
    '''
    gt_lines, gt_names =  read_clusters(images_true_labels,sep,sep2)
    
    mrc = mrcfile.open(images_file_name, mode='r+')
    data = mrc.data
    
    logger.info("Length of data= {}", len(data))
    
    arr_tmp = np.array(data[0])
    logger.info('Image array dimensions= {}', arr_tmp.shape)
    
    mrc.close()

    return data, gt_lines, gt_names