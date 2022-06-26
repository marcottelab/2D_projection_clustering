# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:15:28 2022

@author: Meghana
"""
from argparse import ArgumentParser as argparse_ArgumentParser

import pandas as pd


def get_imp_metric_series(name,meth,dir_nm):
    '''
    Compile important metrics into series from metrics txt file    

    Parameters
    ----------
    name : str
        graph name +'_'+ clustering method
    meth : str
        graph clustering method
    dir_nm : str
        name of results directory

    Returns
    -------
    fin_dict_series: pd.Series
        important metrics and their values 

    '''
    
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
    
    fin_dict_series = pd.Series(fin_dict,name = name)
    
    return fin_dict_series

def main():
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--graph_names", nargs='+', default=['top5_graph','top5_graph_unnorm','all_neigs_graph'], help="Name of slicem graph, specify as list")
    parser.add_argument("--meths", nargs='+', default=['async','greedy_modularity','kclique','label_prop'],help="List of graph clustering methods")    
    parser.add_argument("--dir_nm", default="..\..\\results\synthetic_all\synthetic_graph_clustering", help="Directory containing results")
    args = parser.parse_args()
    
    dir_nm = args.dir_nm
    graph_names = args.graph_names
    meths = args.meths
    
    names = [(graph+'_'+meth,meth) for graph in graph_names for meth in meths]
    
    results_df = pd.DataFrame()
    for name_meth in names:
        series = get_imp_metric_series(name_meth[0],name_meth[1],dir_nm)
        results_df = results_df.append(series)
        
    results_df.to_csv(dir_nm+'\\results.csv')

if __name__ == "__main__":
    main()