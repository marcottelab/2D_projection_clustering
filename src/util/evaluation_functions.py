# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:41:16 2022

@author: Meghana
"""
        
import os
from util.eval_complex import compute_metrics, remove_unknown_prots
from util.input_functions import read_clusters

def evaluate_clusters(clusterwise_indices_start_str,gt_lines,n_clus,clustering_method,out_dir,n_true_clus,gt_names,main_results_dir='../results',suffix='',plot_hist_flag=1,with_junk=1):
    '''
    Evaluate predicted clusters against ground truth
    
    Parameters:    
    clusterwise_indices_start_str (list[tuple(set(string),float)]): List of clusters, each is a tuple of the cluster indices and cluster score (1 by default)
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    n_clus (int): No. of clusters predicted
    clustering_method (sklearn.cluster._method.METHOD): method is a sckit learn clustering method with parameters, 
        ex: DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation(), etc   
    out_dir (string): output directory name for results (will be created if it does not exist)
    n_true_clus (int): No. of true clusters
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order
    
    Returns:
    eval_metrics_dict (dict): Dictionary of evaluation metrics and their values on the predicted set of clusters w.r.t true clusters
    '''    

    out_dir = out_dir + '/evaluate' + suffix
    if not os.path.exists(main_results_dir + '/' + out_dir):
        os.mkdir(main_results_dir + '/' + out_dir)
        
    eval_metrics_dict = dict()
    
    if with_junk:
        eval_metrics_dict = compute_metrics(gt_lines, clusterwise_indices_start_str,main_results_dir + '/' + out_dir + '/' + str(clustering_method),len(gt_lines),n_clus,{"eval_p":0.5,"dir_nm":out_dir},'',gt_names,plot_hist_flag)            
        with open(main_results_dir + '/' + out_dir + '/' + str(clustering_method) + '_metrics.txt', "a") as fid:
            print('No. of predicted clusters = ',n_clus, file=fid)  
            print('No. of true clusters = ',n_true_clus, file=fid)          
            
        eval_metrics_dict["No. of clusters"] = n_clus
            
        if ('FMM F1 score' in eval_metrics_dict) and ('Net F1 score' in eval_metrics_dict) and ('Qi F1 score' in eval_metrics_dict):
            eval_metrics_dict['3 F1 score average'] = (eval_metrics_dict['FMM F1 score'] + eval_metrics_dict['Net F1 score'] + eval_metrics_dict['Qi F1 score'])/3.0

    # After removing unknown projections
    prot_list = set().union(*gt_lines)
    clusterwise_indices_start_str = remove_unknown_prots(clusterwise_indices_start_str, prot_list, 1)    
    n_clus_new = len(clusterwise_indices_start_str)
    eval_metrics_dict_1 = compute_metrics(gt_lines, clusterwise_indices_start_str,main_results_dir + '/' + out_dir + '/' + str(clustering_method),len(gt_lines),n_clus_new,{"eval_p":0.5,"dir_nm":out_dir},'',gt_names,plot_hist_flag)            
    with open(main_results_dir + '/' + out_dir + '/' + str(clustering_method) + '_metrics.txt', "a") as fid:
        print('After removing unknown projections',file=fid)
        print('No. of predicted clusters = ',n_clus_new, file=fid)  
        print('No. of true clusters = ',n_true_clus, file=fid)          
        
    eval_metrics_dict_1["No. of clusters"] = n_clus_new
        
    if ('FMM F1 score' in eval_metrics_dict) and ('Net F1 score' in eval_metrics_dict) and ('Qi F1 score' in eval_metrics_dict):
        eval_metrics_dict_1['3 F1 score average'] = (eval_metrics_dict_1['FMM F1 score'] + eval_metrics_dict_1['Net F1 score'] + eval_metrics_dict_1['Qi F1 score'])/3.0
        
    for key, val in eval_metrics_dict_1.items():
        eval_metrics_dict[key+ ' w/o junk'] = val
            
    if ('FMM F1 score w/o junk' in eval_metrics_dict):
        eval_metrics_dict['FMM F1 score w/o junk']  = eval_metrics_dict.pop('FMM F1 score w/o junk')            

    return eval_metrics_dict



def evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start,main_results_dir='..',file_name = 'slicem_clustering.txt'):
    '''
    Evaluate SLICEM clustering on synthetic dataset
    
    Parameters:    
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order    
    n_true_clus (int): No. of true clusters
    
    Returns:
    eval_metrics_dict (dict): Dictionary of evaluation metrics and their values on the predicted set of clusters w.r.t true clusters
    
    '''
    if dataset == 'synthetic':
        out_dir = 'data/synthetic_dataset'        
    elif dataset == 'synthetic_more_projs_noisy':
        out_dir = 'data/synthetic_more_projs_noisy_dataset'       
    elif dataset == 'synthetic_more_projs':
        out_dir = 'data/synthetic_more_projections'       
    elif dataset == 'synthetic_noisy':
        out_dir = 'data/synthetic_noisy_dataset'           
    else: # real 
        out_dir = 'data/real_dataset'     

    SLICEM_labels_file =  '../' + out_dir + '/' + file_name

    cluster_lines, cluster_numbers =  read_clusters(SLICEM_labels_file,sep)
    n_clus = len(cluster_lines)
    
    cluster_lines = [set([str(int(img_ind) + index_start) for img_ind in entry]) for entry in cluster_lines]
    # Adding score as 1 to satisfy input format for compute_metrics function
    clusterwise_indices_start_str = [(entry,1) for entry in cluster_lines]
    
    eval_metrics_dict = evaluate_clusters(clusterwise_indices_start_str,gt_lines,n_clus,'SLICEM',out_dir,n_true_clus,gt_names,main_results_dir)
    eval_metrics_dict['Silhouette score'] = 'NA'

    
    return eval_metrics_dict


