# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:22:10 2022

@author: Meghana
"""

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
#from networkx.algorithms.community import  naive_greedy_modularity_communities
#from networkx.algorithms.community import louvain_communities
#from networkx.algorithms import community
from networkx.algorithms.community import k_clique_communities, asyn_lpa_communities, label_propagation_communities
# from networkx.algorithms.community import asyn_fluidc
from util.input_functions import get_config, read_clusters
from util.evaluation_functions import evaluate_SLICEM, evaluate_clusters

#import numpy as np
import pandas as pd
import os


def run_clustering(graph,dataset_type,graph_name,graph_type='undirected',main_results_dir='../results'):
    if not nx.is_directed(graph):
        clustering_methods = ['greedy_modularity','kclique','async','label_prop']
    else:
        clustering_methods = ['greedy_modularity','async']

    results_df = pd.DataFrame()

    dataset = dataset_type
    
    images_file_name,images_true_labels,sep,index_start,out_dir_orig, sep2 = get_config(dataset)
    
    out_dir = out_dir_orig 
    
    if not os.path.exists(main_results_dir + '/' + out_dir_orig):
        os.mkdir(main_results_dir + '/' + out_dir_orig)        
    gt_lines, gt_names =  read_clusters(images_true_labels,sep,sep2)
    n_true_clus = len(gt_lines)        
    
    for clustering_method in clustering_methods:    
        if clustering_method == 'greedy_modularity':
            c = greedy_modularity_communities(graph,weight='weight') # 1 community
        elif clustering_method == 'kclique':
            c = list(k_clique_communities(graph,k=2)) # 1 community
        elif clustering_method == 'async':
            c = list(asyn_lpa_communities(graph, weight = 'weight')) # 1 community
        elif clustering_method == 'label_prop':
            c = list(label_propagation_communities(graph)) # 1 community
        
        #c = greedy_modularity_communities(graph,weight='weight',n_communities=4) # No. of communities will not be known, but can cross-validate with a score, so can try this
        #c = list(asyn_fluidc(graph,k=4)) # need to specify no. of communities, can cross-validate
        
        #c = naive_greedy_modularity_communities(graph) # Too slow
        #c = louvain_communities(graph,weight='weight') # Not present in v2.6
        
        # communities_generator = community.girvan_newman(graph)
        # top_level_communities = next(communities_generator)
        # print(sorted(map(sorted, top_level_communities))) # 2 communities, one of which has 1 node
        # next_level_communities = next(communities_generator)
        # print(sorted(map(sorted, next_level_communities))) # 3 communities, 2of which have 1 node each 
        
        n_clus = len(c)
        print(n_clus)
        print(sorted(c[0]))
        
        clusterwise_indices_start_str = [(entry,1) for entry in c]
        


        eval_metrics_dict = evaluate_clusters(clusterwise_indices_start_str,gt_lines,n_clus,clustering_method,out_dir,n_true_clus,gt_names,main_results_dir,suffix='_'+graph_name + '_' + clustering_method+'_'+ graph_type)
        if len(results_df) == 0:
            results_df = pd.DataFrame(columns = eval_metrics_dict.keys())
        results_df = results_df.append(pd.Series(eval_metrics_dict,name = graph_name + ' ' + clustering_method+'_'+ graph_type))
    # print(n_true_clus)
    # print(gt_names)
    # print(gt_lines)

    # if ('MMR F1 score' in eval_metrics_dict):
    #     results_df.sort_values(by='MMR F1 score',ascending=False,inplace=True)        
  
    # results_df.to_csv(main_results_dir + '/' + out_dir_orig + '/graph_clustering_all_methods_sorted_' + dataset + graph_type + '.csv')
    
    return results_df, out_dir_orig, gt_lines,gt_names,n_true_clus,dataset,sep,index_start
    
from argparse import ArgumentParser as argparse_ArgumentParser

parser = argparse_ArgumentParser("Input parameters")
parser.add_argument("--dataset_type", default="real", help="Dataset name, opts: real, synthetic, synthetic_noisy")
parser.add_argument("--graph_name_opts", nargs='+', default=["slicem_edge_list_l2_5_neigs_paper"], help="Name of slicem graph")

args = parser.parse_args()
        
#dataset_type = 'synthetic'
#dataset_type = 'real'
#dataset_type = 'synthetic_noisy'
dataset_type = args.dataset_type


# graph_name = 'all_neigs_graph'
# graph_name = 'slicem_edge_list'
# graph_name = 'slicem_edge_list_l1'
#graph_names = ['slicem_edge_list_l1','slicem_edge_list_euclidean']
#graph_names = ['siamese_l2_5_edge_list','siamese_cosine_5_edge_list','slicem_edge_list_cosine']
#graph_names = ['slicem_edge_list_l2']
graph_names = args.graph_name_opts

#walktrap_cluster_files =['siamese_l2_5_walktrap_clusters.txt','siamese_cosine_5_walktrap_clusters.txt','slicem_cosine_5_walktrap_clusters.txt']
#walktrap_cluster_files =['slicem_clustering.txt']
# walktrap_cluster_files =['slicem_clusters_top3k_l1.txt','slicem_clusters_top3k_l2.txt']

walktrap_cluster_files = ['perfect_clustering.txt','slicem_clusters_top3k_l1.txt','slicem_clusters_top3k_l2.txt','slicem_clustering.txt','slicem_clusters_walktrap_5_l1.txt','slicem_clusters_walktrap_5_euclidean.txt']

# if dataset_type == 'real':
#     dataset = 'real_dataset/slicem_scores_mixture_Euclidean.txt'
# else: # synthetic
#     dataset = 'synthetic_dataset/slicem_scores.txt'

# fname = '../data/' + dataset

# with open(fname) as f:
#     raw_file = [line.rstrip().split() for line in f.readlines()]
    
#     graph_lines = [(line_words[0], line_words[2], line_words[4]) for line_words in raw_file][1:]
  
# graph_str = '\n'.join([' '.join(tup) for tup in graph_lines])

# with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:
#     f.write(graph_str.encode('UTF-8'))

    
results_df_list = []
for graph_name in graph_names:
    graph_type = 'directed'

    with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','rb') as f:    
        graph = nx.read_weighted_edgelist(f, create_using=nx.DiGraph())
        
    # len(graph.edges())
    # Out[8]: 611
        
    df1, out_dir_orig, gt_lines,gt_names,n_true_clus,dataset,sep,index_start = run_clustering(graph,dataset_type,graph_name, graph_type)   

    graph_type = 'undirected'
    with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','rb') as f:    
        graph = nx.read_weighted_edgelist(f)
        
    df2, out_dir_orig, gt_lines,gt_names,n_true_clus,dataset,sep,index_start = run_clustering(graph,dataset_type,graph_name, graph_type)   
    
    results_df_list.append(pd.concat([df1,df2]))
    
results_df = pd.concat(results_df_list)
    
eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start)
results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = 'SLICEM'))

# eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start,main_results_dir='..',file_name = 'slicem_clusters_walktrap_5_euclidean.txt')
# results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = 'SLICEM Euclidean reproduced'))    

# eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start,main_results_dir='..',file_name = 'slicem_clusters_walktrap_5_w_outliers_l1 - Copy.txt')
# results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = 'SLICEM L1 reproduced'))    

for fname in walktrap_cluster_files:
    if fname == 'perfect_clustering.txt':
        index_start = 0
    else:
        index_start = 1    
    eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start,main_results_dir='..',file_name = fname)
    
    results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = fname))    
 
#results_df.sort_values(by='MMR F1 score',ascending=False,inplace=True) 
results_df.sort_values(by='FMM F1 score w/o junk',ascending=False,inplace=True)     

main_results_dir='../results'
results_df.to_csv(main_results_dir + '/' + out_dir_orig + '/graph_clustering_all_methods_sorted_' + dataset_type + '.csv')


# # Constructing graph with 5 nearest neighbors with edge weights as z scores relative to all scores for a given 2D class average
# k=5
# graph_name = 'top5_graph'
# new_graph = nx.Graph()
# # Normalized  
# for node in graph.nodes:
#     neighs = graph[node]
#     '''
#     AtlasView({'1': {'weight': 90.80096628679762}, '2': {'weight': 88.99723972983995}, '
#     '''    
#     neig_weight_tuples = list(zip(neighs.keys(),neighs.values()))
    
#     top_neigs = sorted(neig_weight_tuples,key = lambda x: x[1]['weight'],reverse=True)[:k] # Take top neighbors after getting the scores instead in SLICEM
#     scores = [neig_tup[1]['weight'] for neig_tup in top_neigs]
        
#     mean_score = np.mean(scores)
#     std_dev_score = np.std(scores)
    
#     for neig_tup in top_neigs:
#         neig = neig_tup[0]
#         wt = (neig_tup[1]['weight'] - mean_score)/float(std_dev_score)

#         new_graph.add_edge(node, neig, weight=wt)
        
# with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:    
#     nx.write_weighted_edgelist(new_graph,f)
    
# run_clustering(new_graph,dataset_type,graph_name)    

# # Unnormalized
# graph_name = 'top5_graph_unnorm'
# new_graph = nx.Graph()

# for node in graph.nodes:
#     neighs = graph[node]
#     '''
#     AtlasView({'1': {'weight': 90.80096628679762}, '2': {'weight': 88.99723972983995}, '
#     '''    
#     neig_weight_tuples = list(zip(neighs.keys(),neighs.values()))
    
#     top_neigs = sorted(neig_weight_tuples,key = lambda x: x[1]['weight'],reverse=True)[:k]
#     # scores = [neig_tup[1]['weight'] for neig_tup in top_neigs]
        
#     # mean_score = np.mean(scores)
#     # std_dev_score = np.std(scores)
    
#     for neig_tup in top_neigs:
#         neig = neig_tup[0]
#         #wt = (neig_tup[1]['weight'] - mean_score)/float(std_dev_score)

#         wt = neig_tup[1]['weight'] 
#         new_graph.add_edge(node, neig, weight=wt)
        
# with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:    
#     nx.write_weighted_edgelist(new_graph,f)
    
# run_clustering(new_graph,dataset_type,graph_name)