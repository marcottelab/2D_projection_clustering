# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:22:10 2022

@author: Meghana
"""
from argparse import ArgumentParser as argparse_ArgumentParser
from util.input_functions import get_config, read_clusters
from util.evaluation_functions import evaluate_SLICEM, evaluate_clusters
from networkx.algorithms.community import greedy_modularity_communities, k_clique_communities, asyn_lpa_communities, label_propagation_communities
#from networkx.algorithms.community import  naive_greedy_modularity_communities, louvain_communities, asyn_fluidc
#from networkx.algorithms import community

import networkx as nx
import pandas as pd
import os


def run_clustering(graph,dataset_type,graph_name,graph_type='undirected',main_results_dir='../results'):
    '''
    Graph clustering with different methods - 'kclique','async','label_prop','greedy_modularity' if undirected and 
    'async','greedy_modularity' if directed

    Parameters
    ----------
    graph : networkx graph
        weighted graph
    dataset_type : string
        Name of the dataset
    graph_name : string
        Name of the graph
    graph_type : string, optional
        directed or undirected The default is 'undirected'.
    main_results_dir : string, optional
        Name of the main results directory. The default is '../results'.

    Returns
    -------
    results_df : pandas dataframe
        evaluation metrics comparing predicted complexes with known complexes
    out_dir_orig : string
        Name of the directory with the results
    gt_lines : list[set(string)]
        List of sets of image indices in string format per ground truth cluster
    gt_names : list[string]
        List of cluster names for each cluster in gt_lines in the same order
    n_true_clus : int
        No. of true clusters
    dataset : string
        Name of the dataset
    sep : string
        Separater between name of cluster and its members
    index_start : integer
        Index at which images are started being numbered, ex: 0, 1, etc

    '''
    print(graph_type)
    if not nx.is_directed(graph):
        clustering_methods = ['kclique','async','label_prop','greedy_modularity']
    else:
        clustering_methods = ['async','greedy_modularity']

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
            try:
                c = greedy_modularity_communities(graph,weight='weight') # 1 community
            except:
                continue
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


    # if ('MMR F1 score' in eval_metrics_dict):
    #     results_df.sort_values(by='MMR F1 score',ascending=False,inplace=True)        
  
    # results_df.to_csv(main_results_dir + '/' + out_dir_orig + '/graph_clustering_all_methods_sorted_' + dataset + graph_type + '.csv')
    
    return results_df, out_dir_orig, gt_lines,gt_names,n_true_clus,dataset,sep,index_start
    

def main():
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--dataset_type", default="real", help="Dataset name, opts: real, synthetic, synthetic_noisy")
    parser.add_argument("--graph_name_opts", nargs='+', default=["slicem_edge_list_l2_5_neigs_paper"], help="List of slicem graphs - 'siamese_l2_5_edge_list','siamese_cosine_5_edge_list','slicem_edge_list_cosine', 'all_neigs_graph','slicem_edge_list','slicem_edge_list_l1','slicem_edge_list_euclidean','slicem_edge_list_l2' ")
    parser.add_argument("--walktrap_cluster_files", nargs='+', default=['perfect_clustering.txt','slicem_clusters_top3k_l1.txt','slicem_clusters_top3k_l2.txt','slicem_clustering.txt','slicem_clusters_walktrap_5_l1.txt','slicem_clusters_walktrap_5_euclidean.txt'], help="List of clusters to evaluate, ex: ['siamese_l2_5_walktrap_clusters.txt','siamese_cosine_5_walktrap_clusters.txt','slicem_cosine_5_walktrap_clusters.txt'] ")
    
    
    args = parser.parse_args()
    
    dataset_type = args.dataset_type
    
    graph_names = args.graph_name_opts
    
    
    walktrap_cluster_files = args.walktrap_cluster_files
    
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


if __name__ == "__main__":
    main()