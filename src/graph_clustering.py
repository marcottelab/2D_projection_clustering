# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:22:10 2022

@author: Meghana
"""

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
#from networkx.algorithms.community import  naive_greedy_modularity_communities
#from networkx.algorithms.community import louvain_communities
from networkx.algorithms import community
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import asyn_lpa_communities, label_propagation_communities, asyn_fluidc
from cluster_image_embeddings import get_config, read_clusters, evaluate_clusters
import numpy as np


def run_clustering(graph,dataset_type,graph_name):
    clustering_methods = ['greedy_modularity','kclique','async','label_prop']
    
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
        
        dataset = dataset_type
        
        images_file_name,images_true_labels,sep,index_start,out_dir_orig = get_config(dataset)
        
        out_dir = out_dir_orig 
        gt_lines, gt_names =  read_clusters(images_true_labels,sep)
        n_true_clus = len(gt_lines)
        eval_metrics_dict = evaluate_clusters(clusterwise_indices_start_str,gt_lines,n_clus,clustering_method,out_dir,n_true_clus,gt_names,main_results_dir='../results',suffix='_'+graph_name + '_' + clustering_method)
  
        
dataset_type = 'synthetic'
#dataset_type = 'real'
if dataset_type == 'real':
    dataset = 'real_dataset/slicem_scores_mixture_Euclidean.txt'
else: # synthetic
    dataset = 'synthetic_dataset/slicem_scores.txt'

fname = '../data/' + dataset
graph_name = 'all_neigs_graph'

with open(fname) as f:
    raw_file = [line.rstrip().split() for line in f.readlines()]
    
    graph_lines = [(line_words[0], line_words[2], line_words[4]) for line_words in raw_file][1:]
  
graph_str = '\n'.join([' '.join(tup) for tup in graph_lines])

with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:
    f.write(graph_str.encode('UTF-8'))

with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','rb') as f:    
    graph = nx.read_weighted_edgelist(f)
    
run_clustering(graph,dataset_type,'all_neigs_graph')   
    
# Constructing graph with 5 nearest neighbors with edge weights as z scores relative to all scores for a given 2D class average
k=5
graph_name = 'top5_graph'
new_graph = nx.Graph()
# Normalized  
for node in graph.nodes:
    neighs = graph[node]
    '''
    AtlasView({'1': {'weight': 90.80096628679762}, '2': {'weight': 88.99723972983995}, '
    '''    
    neig_weight_tuples = list(zip(neighs.keys(),neighs.values()))
    
    top_neigs = sorted(neig_weight_tuples,key = lambda x: x[1]['weight'],reverse=True)[:k]
    scores = [neig_tup[1]['weight'] for neig_tup in top_neigs]
        
    mean_score = np.mean(scores)
    std_dev_score = np.std(scores)
    
    for neig_tup in top_neigs:
        neig = neig_tup[0]
        wt = (neig_tup[1]['weight'] - mean_score)/float(std_dev_score)

        new_graph.add_edge(node, neig, weight=wt)
        
with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:    
    nx.write_weighted_edgelist(new_graph,f)
    
run_clustering(new_graph,dataset_type,graph_name)    

# Unnormalized
graph_name = 'top5_graph_unnorm'
new_graph = nx.Graph()

for node in graph.nodes:
    neighs = graph[node]
    '''
    AtlasView({'1': {'weight': 90.80096628679762}, '2': {'weight': 88.99723972983995}, '
    '''    
    neig_weight_tuples = list(zip(neighs.keys(),neighs.values()))
    
    top_neigs = sorted(neig_weight_tuples,key = lambda x: x[1]['weight'],reverse=True)[:k]
    # scores = [neig_tup[1]['weight'] for neig_tup in top_neigs]
        
    # mean_score = np.mean(scores)
    # std_dev_score = np.std(scores)
    
    for neig_tup in top_neigs:
        neig = neig_tup[0]
        #wt = (neig_tup[1]['weight'] - mean_score)/float(std_dev_score)

        wt = neig_tup[1]['weight'] 
        new_graph.add_edge(node, neig, weight=wt)
        
with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:    
    nx.write_weighted_edgelist(new_graph,f)
    
run_clustering(new_graph,dataset_type,graph_name)