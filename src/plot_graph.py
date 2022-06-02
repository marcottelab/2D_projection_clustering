# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:57:13 2022

@author: Meghana
"""
import networkx as nx

G = nx.read_weighted_edgelist('../data/real_dataset/slicem_edge_list_l1_paper.txt')

G.remove_edge('43','54')

comps = list(nx.connected_components(G))
print(comps)

nx.draw(G,with_labels=True)

with open('../data/real_dataset/l1_paper_clusters_rerun.txt','w') as f:
    for i,comp in enumerate(comps):
        f.write(str(i)+'   ['+', '.join(comp)+']\n')