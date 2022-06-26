# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:24:13 2021

@author: Meghana
"""
from numpy import random as rnd

import numpy as np
import networkx as nx


def f1_fmm(wt_mat):
    '''
    Get FMM F1 score

    Parameters
    ----------
    wt_mat : numpy.ndarray
        Adjacency matrix of similarities between true and predicted complexes

    Returns
    -------
    prec : float
        FMM precision
    recall : float
        FMM recall
    f1 : float
        FMM F1 score
    max_matching_edges : dict
        true complex to predicted complex and vice-versa matches, ex: {'t2': 'p3', 't1': 'p2', 't0': 'p0', 'p3': 't2'} 

    '''
    wt_mat = - wt_mat
    
    nr, nc = np.shape(wt_mat)
    
    B = nx.Graph()
    
    # Add nodes with the node attribute "bipartite"
    
    B.add_nodes_from(['t' + str(i) for i in range(nr)], bipartite=0)
    
    B.add_nodes_from(['p' + str(j) for j in range(nc)], bipartite=1)
    
    # Add edges only between nodes of opposite node sets
    
    B.add_weighted_edges_from([('t' + str(i),'p' + str(j), wt_mat[i,j]) for i in range(nr) for j in range(nc)])
    
    
    max_matching_edges = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    
    sum_wts = -sum([B[key][val]['weight'] for key,val in max_matching_edges.items()])/2
    
    recall = sum_wts/nr
    
    prec = sum_wts/nc
    
    f1 = 2*recall*prec/(prec+recall)
    
    return prec, recall, f1, max_matching_edges


def test_f1_fmm():
    wt_mat = rnd.rand(3,4)
    prec, recall, f1, max_matching_edges = f1_fmm(wt_mat)
    print(f1)
    print(max_matching_edges)
    print(wt_mat)
    
if __name__ == "__main__":
    test_f1_fmm()
