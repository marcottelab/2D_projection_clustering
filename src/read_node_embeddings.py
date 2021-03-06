# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:24:48 2022

@author: Meghana
"""
from pickle import load as pkl_load

import numpy as np


def slicem_graph_embeddings(dataset, node_embedding_method='metapath2vec', implementation = 'stellar',graph = 'slicem_edge_list',embedding_to_combine='siamese',graph_type=''):
    '''
    Get graph embeddings

    Parameters
    ----------
    dataset : string
        Name of the dataset
    node_embedding_method : string, optional
        Name of the node embedding method. The default is 'metapath2vec'.
    implementation : string, optional
        stellar or original node2vec. The default is 'stellar'.
    graph : string, optional
        The name of the graph. The default is 'slicem_edge_list'.
    embedding_to_combine : string, optional
        Name of the embedding to combine. The default is 'siamese'.
    graph_type : string, optional
        Type of graph - directed or undirected as mentioned in the file name. The default is ''.

    Returns
    -------
    vectors : numpy.ndarray
        Array of image embeddings.

    '''
    
    if implementation == 'original':
        if graph == 'full':
            fname = './node2vec-master/emb/real_graph_embeddings.txt'
        else:
            fname = './node2vec-master/emb/top5_graph_unnorm_embeddings.txt'
            
        with open(fname) as f:
            raw_lines = f.readlines()[1:]
            
            lines = [(int(line.split()[0]),line.split()[1:]) for line in raw_lines]
            
        sorted_lines = sorted(lines,key = lambda x: x[0])
        
        array = [line[1] for line in sorted_lines]
        array_nums = [[float(str_num) for str_num in line] for line in array]
            
        vectors = np.array(array_nums)
        
    else: #  stellargraph
        if dataset == "synthetic_more_projs_wo_4v6c":
            dataset = "synthetic_more_projs_wo_4v6c"+ '/node_embs/'
        else:
            dataset = dataset + '_dataset' + '/graph_embeddings/'
    
        with open('../data/' + dataset +graph+graph_type+'_stellar_' + node_embedding_method + embedding_to_combine+ '.npy', 'rb') as f:
            node_embeddings = np.load(f)
            
        with open('../data/' + dataset + graph+graph_type+'_stellar_' + node_embedding_method + embedding_to_combine+ '_node_ids.list', 'rb') as f:
            node_ids = pkl_load(f)
        
        int_node_ids = [int(node) for node in node_ids]
        
        node_tup = list(zip(node_embeddings,int_node_ids))
        
        sorted_node_tup = sorted(node_tup,key = lambda x: x[1])
        
        vectors = np.array([node_emb_id[0] for node_emb_id in sorted_node_tup])
    
    return vectors