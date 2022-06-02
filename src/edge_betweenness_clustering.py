# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:35:43 2022

@author: Meghana
"""
from igraph import Graph

graph_names = ['slicem_edge_list_l1_5_neigs_paper_disc']
#graph_names = ['slicem_edge_list_top3k_l1','slicem_edge_list_l2_top3k','slicem_edge_list_euclidean','slicem_edge_list_l1','slicem_edge_list_l1_5_neigs_paper','slicem_edge_list_l2_5_neigs_paper']
#graph_names = ['slicem_edge_list_l1_5_neigs_paper','slicem_edge_list_l2_5_neigs_paper']

#method = 'edge_betweenness'
#method = 'cc'
#method = 'cc_strong'
#method = 'edge_betweenness_wt'
#method = 'walktrap'

methods = ['walktrap','edge_betweenness_wt','cc_strong','cc','edge_betweenness']

def get_index_name_map(g):
    # Get index to name map 
    index_name_map = dict()
    
    for i, node in enumerate(list(g.vs())):
        index_name_map[i] = node["name"]
        
    return index_name_map


for graph_name in graph_names:
    graph_path = '../data/real_dataset/'+ graph_name + '.txt'
    for method in methods:
        communities_out_path = '../data/real_dataset/'+ graph_name + method + '_communities'+'.txt'
        
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
            # # Use edge betweenness to detect communities
            # communities = g.community_edge_betweenness(directed=True,weights='weight')        
            # # ... and convert into a VertexClustering for plotting
            # try:
            #     communities = communities.as_clustering()        
            # except:
            #     # Try clustering on connected components
            #     try:
            #         # Get CC 
            #         cc_s = g.clusters(mode='weak')
            #         communities = []
            #         for comm in cc_s:
            #             commun = comm.community_edge_betweenness(directed=True,weights='weight')        
            #             communities1 = commun.as_clustering()  
            #             communities = communities + list(communities1)
                    
            #     except:
            #         print('Error with: ' + graph_name)                
            #         continue
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