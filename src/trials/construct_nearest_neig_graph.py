# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:47:05 2022

@author: Meghana
"""
from graph_clustering_nx import run_clustering

import networkx as nx
import numpy as np

graph_name = 'all_neigs'
dataset_type = 'real'

if dataset_type == 'real':
    dataset = 'real_dataset/slicem_scores_mixture_Euclidean.txt'
else: # synthetic
    dataset = 'synthetic_dataset/slicem_scores.txt'

fname = '../data/' + dataset

with open(fname) as f:
    raw_file = [line.rstrip().split() for line in f.readlines()]
    
    graph_lines = [(line_words[0], line_words[2], line_words[4]) for line_words in raw_file][1:]
  
graph_str = '\n'.join([' '.join(tup) for tup in graph_lines])

with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','wb') as f:
    f.write(graph_str.encode('UTF-8'))
    
with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','rb') as f:    
    graph = nx.read_weighted_edgelist(f, create_using=nx.DiGraph())

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
    
    top_neigs = sorted(neig_weight_tuples,key = lambda x: x[1]['weight'],reverse=True)[:k] # Take top neighbors after getting the scores instead in SLICEM
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