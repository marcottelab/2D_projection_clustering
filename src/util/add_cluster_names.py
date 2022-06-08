# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:15:07 2022

@author: Meghana
"""

with open('../../data/synthetic_dataset/synthetic_2D_order.txt') as f:
    lines = f.readlines()
    
names = [name.split()[0] for name in lines][1:]

with open('../../data/synthetic_dataset/synthetic_true_clustering_wo_names.txt') as f:
    lines = f.readlines()
    
clusters = [name.rstrip().split('\t')[1] for name in lines]

with open('../../data/synthetic_dataset/synthetic_true_clustering.txt','w') as f:
    f.writelines(['\t'.join(entry)+'\n' for entry in list(zip(names,clusters))])