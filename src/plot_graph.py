# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:57:13 2022

@author: Meghana
"""
from argparse import ArgumentParser as argparse_ArgumentParser

import networkx as nx

def main():
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--graph_path", default="../data/real_dataset/slicem_edge_list_l1_paper.txt", help="Path to graph")
    parser.add_argument("--rm_edge_and_save_connected_components", default=True,help="If using real graph - slicem_edge_list_l1_paper.txt with edge 43-54 removing which separates graph into 3 components")
    
    args = parser.parse_args()   
    
    G = nx.read_weighted_edgelist(args.graph_path)
    nx.draw(G,with_labels=True)    
    
    if args.rm_edge_and_save_connected_components:
        G.remove_edge('43','54')
        
        comps = list(nx.connected_components(G))
        print(comps)
        
        nx.draw(G,with_labels=True)
        
        with open('../data/real_dataset/l1_paper_clusters_rerun.txt','w') as f:
            for i,comp in enumerate(comps):
                f.write(str(i)+'   ['+', '.join(comp)+']\n')
            