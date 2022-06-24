# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 19:27:11 2022

@author: Meghana
"""

from glob import glob
import pandas as pd

datasets = ['real_all','synthetic_more_projs_all','synthetic_all','synthetic_noisy_all','synthetic_more_projs_noisy_all','synthetic_more_projs_wo_4v6c_all']

level= 1

for dataset in datasets:

    if dataset in ['real_all','synthetic_all']:
        level = 2
        
    if level == 1:
        allsubd = ['../results/'+dataset+'/*/*.csv']# gets all nested subdirectories
    elif level == 2:
        allsubd1 = '../results/'+dataset+'/*/*.csv' # gets all nested subdirectories
        allsubd2 = '../results/'+dataset+'/*/*/*.csv' # gets all nested subdirectories
        allsubd = [allsubd1,allsubd2]
    else:
        allsubd = ['../results/'+dataset+'/*/*/*.csv']
    
    fnames = []
    
    for subd in allsubd:
        fnames = fnames + glob(subd, recursive=True)
        
    fnames_all = [fname for fname in fnames if ('hyper' not in fname) and ('test' not in fname) and ('embedding' not in fname) and ('evaluating' not in fname)]
    
    df_list = []
    rename_map = {'No. of matches in MMR':'No. of matches in FMM','MMR F1 score': 'FMM F1 score','MMR Precision': 'FMM Precision','MMR Recall': 'FMM Recall','Net F1 score':'CMFF'}
    for fname in fnames_all:
        df_tmp = pd.read_csv(fname,index_col=0)
        df_tmp['name'] = fname
        df_tmp = df_tmp.rename(columns={k: v for k, v in rename_map.items() if k in df_tmp})
        df_list.append(df_tmp)
        
    df_compiled = pd.concat(df_list)
    
    df_compiled = df_compiled.sort_values(by='FMM F1 score',ascending=False)
    
    df_compiled.to_csv('../results/' + dataset + '/'+dataset+'_all_results_compiled.csv')
        