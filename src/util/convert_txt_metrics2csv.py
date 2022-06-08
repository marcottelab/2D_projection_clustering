# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:15:28 2022

@author: Meghana
"""
import pandas as pd

def get_imp_metric_series(name,meth,dir_nm):
    
    fname = dir_nm + "\evaluate_"+ name + "\\" + meth +"_metrics.txt"

    with open(fname) as f:
        metric_lines = f.readlines()
        
    name_value_pairs = [line.rstrip().split(' = ') for line in metric_lines]
    
    pair_dict = dict([pair for pair in name_value_pairs if len(pair) == 2])
    
    pair_dict['Qi Precision'] = pair_dict['Prediction Precision']
    pair_dict['Qi Recall'] = pair_dict['Prediction Recall']
    pair_dict['Qi F1 score'] = pair_dict['Prediction F1 score']
    pair_dict['No. of clusters'] = pair_dict['No. of predicted clusters']
    
    imp_metrics = ['No. of matches in MMR','MMR Precision','MMR Recall','MMR F1 score','Net F1 score','Qi Precision','Qi Recall','Qi F1 score','No. of clusters']
    fin_dict = {key: pair_dict[key] for key in imp_metrics}
    
    return pd.Series(fin_dict,name = name)    

dir_nm = "..\..\\results\synthetic_all\synthetic_graph_clustering"
graph_names = ['top5_graph','top5_graph_unnorm','all_neigs_graph']
meths = ['async','greedy_modularity','kclique','label_prop']
names = [(graph+'_'+meth,meth) for graph in graph_names for meth in meths]

results_df = pd.DataFrame()
for name_meth in names:
    series = get_imp_metric_series(name_meth[0],name_meth[1],dir_nm)
    results_df = results_df.append(series)
    
results_df.to_csv(dir_nm+'\\results.csv')
    

