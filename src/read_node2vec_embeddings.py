# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:24:48 2022

@author: Meghana
"""


import numpy as np

def slicem_node2vec_graph_embeddings():
    #fname = './node2vec-master/emb/real_graph_embeddings.txt'
    
    fname = './node2vec-master/emb/top5_graph_unnorm_embeddings.txt'
        
    with open(fname) as f:
        raw_lines = f.readlines()[1:]
        
        lines = [(int(line.split()[0]),line.split()[1:]) for line in raw_lines]
        
    sorted_lines = sorted(lines,key = lambda x: x[0])
    
    array = [line[1] for line in sorted_lines]
    array_nums = [[float(str_num) for str_num in line] for line in array]
        
    vectors = np.array(array_nums)
    
    return vectors