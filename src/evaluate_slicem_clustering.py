# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:16:28 2022

@author: Meghana
"""
from util.input_functions import get_config, read_clusters
from util.evaluation_functions import evaluate_SLICEM

import pandas as pd


dataset = 'real'
images_file_name,images_true_labels,sep,index_start,out_dir_orig, sep2 = get_config(dataset)

gt_lines, gt_names =  read_clusters(images_true_labels,sep,sep2)
n_true_clus = len(gt_lines)
results_df = pd.DataFrame()

fnames = ['l1_paper_clusters_rerun_apo.txt','slicem_clustering_3.txt','l1_paper_clusters_rerun.txt','perfect_clustering.txt','slicem_clusters_top3k_l1.txt','slicem_clusters_top3k_l2.txt','slicem_clustering.txt','slicem_clusters_walktrap_5_l1.txt','slicem_clusters_walktrap_5_euclidean.txt']
graph_names = ['slicem_edge_list_l1_5_neigs_paper_disc','slicem_edge_list_top3k_l1','slicem_edge_list_l2_top3k','slicem_edge_list_euclidean','slicem_edge_list_l1','slicem_edge_list_l1_5_neigs_paper','slicem_edge_list_l2_5_neigs_paper']

methods = ['edge_betweenness_wt','edge_betweenness','cc','cc_strong','walktrap']

for method in methods:
    for graph_name in graph_names:
        fnames.append(graph_name + method + '_communities'+'.txt')

for fname in fnames:
    if fname == 'perfect_clustering.txt':
        index_start = 0
    else:
        index_start = 1
    eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start,main_results_dir='..',file_name = fname)
    results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = fname))

results_df.sort_values(by='MMR F1 score w/o junk',ascending=False,inplace=True)   

results_df.to_csv('../data/real_dataset/slicem_clustering_eval_all.csv')