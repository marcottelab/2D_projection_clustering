# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:35:43 2022

@author: Meghana
"""
from igraph import Graph
from argparse import ArgumentParser as argparse_ArgumentParser


def get_index_name_map(g):
    '''
    Get index to name map 

    Parameters
    ----------
    g : igraph graph
        similarity weighted graph

    Returns
    -------
    index_name_map : dict
        map of graph node indices to graph node names

    '''
    index_name_map = dict()
    
    for i, node in enumerate(list(g.vs())):
        index_name_map[i] = node["name"]
        
    return index_name_map


def main():
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--dataset_type", default="real", help="Dataset name, opts: real, synthetic, synthetic_noisy")
    parser.add_argument("--graph_name_opts", nargs='+', default=['slicem_edge_list_l1_5_neigs_paper_disc'], help="Name of slicem graph, ex: ['slicem_edge_list_top3k_l1','slicem_edge_list_l2_top3k','slicem_edge_list_euclidean','slicem_edge_list_l1','slicem_edge_list_l1_5_neigs_paper','slicem_edge_list_l2_5_neigs_paper']")
    parser.add_argument("--clustering_meths", nargs='+', default=['edge_betweenness_wt','edge_betweenness','cc','cc_strong','walktrap'],help="Clustering methods")
    
    args = parser.parse_args()
      
    graph_names = args.graph_name_opts
    
    methods = args.clustering_meths
    
    for graph_name in graph_names:
        graph_path = '../data/'+args.dataset_type+'_dataset/'+ graph_name + '.txt'
        for method in methods:
            communities_out_path = '../data/'+args.dataset_type+'_dataset/'+ graph_name + method + '_communities'+'.txt'
            
            g=Graph.Read_Ncol(graph_path,names=True)
            
            # Get index to name map 
            index_name_map = get_index_name_map(g)
            
            communities_w_names = None
            
            if method == 'edge_betweenness':   
                # Use edge betweenness to detect communities
                communities = g.community_edge_betweenness(directed=True)
                # ... and convert into a VertexClustering for plotting
                communities = communities.as_clustering()
            elif method == 'edge_betweenness_wt':   
                
                cc_s = g.clusters(mode='weak')
                communities_w_names = []
                for comm in cc_s:
                    subg = g.subgraph(comm).copy()
                    commun = subg.community_edge_betweenness(directed=True,weights='weight')        
                    communities1 = commun.as_clustering()
        
                    index_name_map1 = get_index_name_map(subg)
                    communities_w_names1 = [[index_name_map1[node] for node in comm1] for comm1 in list(communities1)]
                    #print(list(communities_w_names1))            
                    communities_w_names = communities_w_names + list(communities_w_names1)        
    
            elif method == 'cc_strong':
                communities = g.clusters()
            elif method == 'walktrap':
                wt = Graph.community_walktrap(g, weights='weight', steps=4)
                communities = wt.as_clustering()        
            else:
                communities = g.clusters(mode='weak')
                
            # Remove outliers and keep them as a separate cluster
            if not communities_w_names:
                communities_w_names = [[index_name_map[node] for node in comm] for comm in list(communities)]
            
            with open(communities_out_path,'w') as f:
                for i, comm in enumerate(communities_w_names):
                    f.write(str(i) + '   [' + ', '.join(comm)+']\n')