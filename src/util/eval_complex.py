# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:24 2019

@author: Meghana
"""
from seaborn import displot as sns_displot
from numpy import zeros as np_zeros, count_nonzero as np_count_nonzero, sum as np_sum, argmax as np_argmax, sqrt as np_sqrt
from logging import info as logging_info
from matplotlib.pyplot import figure as plt_figure, savefig as plt_savefig, close as plt_close, xlabel as plt_xlabel, title as plt_title, plot as plt_plot,ylabel as plt_ylabel, rc as plt_rc, rcParams as plt_rcParams
from util.convert_humap_ids2names import convert2names_wscores_matches
from collections import Counter
from util.test_F1_FMM import f1_fmm


def write_best_matches(best_matches_for_known,out_comp_nm,dir_nm,suffix,write_comp_score=0,gt_names=[]):
    '''
    Write best matches for known complexes using CMMF
    
    Parameters
    ----------
    best_matches_for_known : list[tuple(set(str),set(str),float,float]
        Predicted complex nodes, known complex nodes, metric score, complex score
    out_comp_nm : str
        Output directory name
    dir_nm : str
        Directory name correspondig to experiment, ex: hu.MAP
    suffix : str
        suffix for results: train, test or both 
    write_comp_score : bool, optional
        Write complex score or not The default is 0.
    gt_names : list[string], optional
        List of cluster names for each cluster in gt_lines in the same orders. The default is [].

    Returns
    -------
    None.

    '''

    if len(gt_names) == len(best_matches_for_known):
        sorted_matches_zip = sorted(list(zip(best_matches_for_known,gt_names)),key=lambda x: x[0][2],reverse=True)
        sorted_matches = [entry[0] for entry in sorted_matches_zip]
        known_comp_names = [entry[1] for entry in sorted_matches_zip]
        name_flag = 1
    else:
        sorted_matches = sorted(best_matches_for_known,key=lambda x: x[2],reverse=True)
        name_flag = 0
    if dir_nm == "humap":
        convert2names_wscores_matches(sorted_matches, out_comp_nm + suffix + '_known_pred_matches_names.txt')
 
    with open(out_comp_nm + suffix + '_known_pred_matches.txt', "w") as fn:
        fn_write = fn.write
        if write_comp_score:
            if name_flag:
                fn_write("Known complex ||| Known complex nodes ||| Best match Predicted complex nodes ||| Match F1 score ||| Complex score \n")
            else:
                fn_write("Known complex nodes ||| Best match Predicted complex nodes ||| Match F1 score ||| Complex score \n")
        else:
            if name_flag:
                fn_write("Known complex ||| Known complex nodes ||| Best match Predicted complex nodes ||| Match F1 score \n")
            else:                
                fn_write("Known complex nodes ||| Best match Predicted complex nodes ||| Match F1 score \n")
        

        for index in range(len(sorted_matches)):
            try:
                pred_graph_nodes = sorted(list(sorted_matches[index][0]),key = lambda x: int(x))
                known_graph_nodes = sorted(list(sorted_matches[index][1]),key = lambda x: int(x))
            except:    
                pred_graph_nodes = sorted(list(sorted_matches[index][0]))
                known_graph_nodes = sorted(list(sorted_matches[index][1]))
            match_score = sorted_matches[index][2]
            complex_score = sorted_matches[index][3]
            if name_flag:
                fn_write("%s " % known_comp_names[index])
                fn_write(" ||| ")
            for node in known_graph_nodes:
                fn_write("%s " % node)
            fn_write(" ||| ")
            for node in pred_graph_nodes:
                fn_write("%s " % node)
            
            fn_write(" ||| ")
            fn_write("%.3f" % match_score)
            if write_comp_score:
                fn_write(" ||| ")
                fn_write("%.3f" % float(complex_score))            
            fn_write("\n")


def write_best_matches_best4pred(best_matches_for_known,out_comp_nm,dir_nm,suffix,write_comp_score=0):
    '''
    Write best matches for predicted complexes using CMMF

    Parameters
    ----------
    best_matches_for_known : list[tuple(set(str),set(str),float,float]
        Predicted complex nodes, known complex nodes, metric score, complex score
    out_comp_nm : str
        Output directory name
    dir_nm : str
        Directory name corresponding to experiment, ex: hu.MAP
    suffix : str
        suffix for results: train, test or both 
    write_comp_score : bool, optional
        Write complex score or not The default is 0.

    Returns
    -------
    None.

    '''
    if len(best_matches_for_known[0]) == 5:
        name_flag = 1
    else:
        name_flag = 0
       
    sorted_matches = sorted(best_matches_for_known,key=lambda x: x[2],reverse=True)
    if dir_nm == "humap":
        convert2names_wscores_matches(sorted_matches, out_comp_nm + suffix + '_known_pred_matches_names.txt')
 
    with open(out_comp_nm + suffix + '_known_pred_matches.txt', "w") as fn:
        fn_write = fn.write
        if write_comp_score:
            if not name_flag:
                fn_write("Predicted complex nodes||| Best match Known complex nodes ||| Match F1 score ||| Complex score \n")
            else:
                fn_write("Predicted complex nodes||| Best match Known complex nodes ||| Best match Known complex ||| Match F1 score ||| Complex score \n")
                
        else:
            if not name_flag:
                fn_write("Predicted complex nodes ||| Best match Known complex nodes ||| Match F1 score \n")
            else:
                fn_write("Predicted complex nodes ||| Best match Known complex nodes ||| Best match Known complex ||| Match F1 score \n")

        for index in range(len(sorted_matches)):
            try:
                pred_graph_nodes = sorted(list(sorted_matches[index][0]),key = lambda x: int(x))
                known_graph_nodes = sorted(list(sorted_matches[index][1]),key = lambda x: int(x))
            except:    
                pred_graph_nodes = sorted(list(sorted_matches[index][0]))
                known_graph_nodes = sorted(list(sorted_matches[index][1]))
            match_score = sorted_matches[index][2]
            complex_score = sorted_matches[index][3]
            for node in pred_graph_nodes:
                fn_write("%s " % node)
            fn_write(" ||| ")
            for node in known_graph_nodes:
                fn_write("%s " % node)
            fn_write(" ||| ")    
            if name_flag:
                fn_write("%s " % sorted_matches[index][4])
            
            fn_write(" ||| ")
            fn_write("%.3f" % match_score)
            if write_comp_score:            
                fn_write(" ||| ")
                fn_write("%.3f" % float(complex_score))            
            fn_write("\n")
            
            
def plot_f1_scores(best_matches,out_comp_nm,suffix,prefix,plot_hist_flag=1):
    '''
    
    Plot histogram of F1 scores

    Parameters
    ----------
    best_matches : list[tuple(set(str),set(str),float,float]
        Predicted complex nodes, known complex nodes, metric score, complex score
    out_comp_nm : str
        Output directory name
    suffix : str
        suffix for results: train, test or both 
    prefix : str
        Best match for known or best match for predicted
    plot_hist_flag : bool, optional
        Plot F1 score histograms of best matches or not. The default is 1.

    Returns
    -------
    avged_f1_score : float
        Average F1 score

    '''
    max_f1_scores = [match[2] for match in best_matches]
    
    avged_f1_score = sum(max_f1_scores)/float(len(max_f1_scores))
    
    f1_score_counts = Counter()
    
    for score in max_f1_scores:
        f1_score_counts[score] += 1
        
    n_perfect_matches = 0
    if 1 in f1_score_counts:
        n_perfect_matches = f1_score_counts[1]
        
    n_no_matches = 0
    if 0 in f1_score_counts:
        n_no_matches = f1_score_counts[0]    
                
    if len(set(max_f1_scores)) > 1:
        if plot_hist_flag:
            fig = plt_figure(figsize=(12,10))
            plt_rcParams["font.family"] = "Times New Roman"
            plt_rcParams["font.size"] = 16
            sns_displot(max_f1_scores)
            plt_xlabel("F1 score")
            plt_ylabel('Frequency')
            plt_title(prefix + "F1 score distribution")
            #plt_savefig(out_comp_nm +suffix+ '_f1_scores_histogram.eps',dpi=350,format='eps')
            plt_savefig(out_comp_nm +suffix+ '_f1_scores_histogram.tiff',dpi=350,format='tiff')
            plt_savefig(out_comp_nm +suffix+ '_f1_scores_histogram.jpg',dpi=350,format='jpg')
            
            plt_close(fig)    
        
    with open(out_comp_nm + '_metrics.txt', "a") as fid:
        print(prefix, file=fid)
        print("Averaged F1 score = %.3f" % avged_f1_score, file=fid)
        print("No. of perfectly recalled matches = %d" % n_perfect_matches, file=fid)
        print("No. of matches not recalled at all = %d" % n_no_matches, file=fid)     
    return avged_f1_score
            
    
def plot_pr_curve_FMM(Metric,fin_list_graphs,out_comp_nm):
    '''
    Plot PR curve using FMM metric and individual complex score thresholds

    Parameters
    ----------
    Metric : numpy.ndarray
        Matrix of match scores between test and predicted complexes
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores. 
    out_comp_nm : str
        Output directory name

    Returns
    -------
    None.

    '''
    
    n_divs = 10
    scores_list = [float(pred_complex[1]) for pred_complex in fin_list_graphs]
    #print(scores_list)
    min_score = min(scores_list)
    interval_len = (max(scores_list) - min_score)/float(n_divs)
    thresholds = [min_score + i*interval_len for i in range(n_divs)]
    
    precs = []
    recalls = []
    for thres in thresholds:
        # list of indices with scores greater than the threshold
        col_inds = [j for j,score in enumerate(scores_list) if score >= thres]
        prec_FMM, recall_FMM, f1_FMM, max_matching_edges = f1_fmm(Metric[:,col_inds])
        
        precs.append(prec_FMM)
        recalls.append(recall_FMM)
        
    fig = plt_figure()
    plt_plot(recalls,precs,'.-')
    plt_ylabel("Precision")
    plt_xlabel("Recall")
    plt_title("PR curve for FMM measure")
    plt_savefig(out_comp_nm + '_pr_FMM.png')
    plt_close(fig)    
    

def f1_similarity(P,T):
    '''
    Calculate f1 similarity measure between a predicted and a known complex

    Parameters
    ----------
    P : set(str)
        Predicted complex nodes
    T : set(str)
        True complex nodes

    Returns
    -------
    F1_score: float
        F1 similarity score
    C : set(str)
        Common nodes between predicted and known complexes

    '''
    C = len(T.intersection(P))
    
    if len(P) == 0 or len(T) == 0:
        return 0, C 
    
    Precision = float(C) / len(P)
    Recall = float(C) / len(T)
    
    if Precision == Recall == 0:
        F1_score = 0
    else:
        F1_score = 2 * Precision * Recall / (Precision + Recall)   
        
    return F1_score, C 


def one2one_matches(known_complex_nodes_list, fin_list_graphs, N_pred_comp, N_test_comp,out_comp_nm,suffix,dir_nm,plot_pr_flag=0,gt_names=[],plot_hist_flag=1):
    '''
    Compute evaluation metrics - FMMF, CMMF, UnSPA

    Parameters
    ----------
    known_complex_nodes_list : list[list[str]]
        List of list of nodes (proteins) in each known complex.
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores
    N_pred_comp : int
        No. of predicted complexes
    N_test_comp : int
        No. of test complexes
    out_comp_nm : str
        Output directory name
    suffix : str
        suffix for results: train, test or both 
    dir_nm : str
        Directory name correspondig to experiment, ex: hu.MAP
    plot_pr_flag : bool, optional
        Plot PR curve or notThe default is False.        
    gt_names : TYPE(list[string]), optional
        DESCRIPTION. List of cluster names for each cluster in gt_lines in the same orders. The default is [].
    plot_hist_flag : bool, optional
        Plot F1 score histograms of best matches or not. The default is 1.

    Returns
    -------
    avg_f1_score: float
        Average of average F1 scores of best matches for known and best matches for predicted
    net_f1_score: float
        CMMF
    PPV: float
        Unbiased PPV
    Sn: float
        Unbiased Sn
    acc_unbiased: float
        UnSPA
    prec_FMM: float
        FMM precision
    recall_FMM: float
        FMM recall
    f1_FMM: float
        FMM F1 score
    n_matches: int
        No. of matches in FMMF

    '''
    Metric = np_zeros((N_test_comp, N_pred_comp))
    Common_nodes = np_zeros((N_test_comp, N_pred_comp))
    known_comp_lens = np_zeros((N_test_comp, 1))
    pred_comp_lens = np_zeros((1, N_pred_comp))
    
    fl = 1

    for i, test_complex in enumerate(known_complex_nodes_list):
        T = set(test_complex)
        known_comp_lens[i,0] = len(T)
        
        for j, pred_complex in enumerate(fin_list_graphs):
            P = pred_complex[0]
            
            F1_score, C = f1_similarity(P,T)
            Common_nodes[i, j] = C
            
            Metric[i, j] = F1_score
            
            if fl == 1:
                pred_comp_lens[0,j] = len(P)
        fl = 0
        
    max_indices_i_common = np_argmax(Common_nodes, axis=0)
    ppv_list = [ float(Common_nodes[i,j])/pred_comp_lens[0,j] for j,i in enumerate(max_indices_i_common)]
    PPV = sum(ppv_list)/len(ppv_list)
    
    max_indices_j_common = np_argmax(Common_nodes, axis=1)
    sn_list = [ float(Common_nodes[i,j])/known_comp_lens[i,0] for i,j in enumerate(max_indices_j_common)]
    Sn = sum(sn_list)/len(sn_list)
    
    acc_unbiased = np_sqrt(PPV * Sn)

    max_indices_i = np_argmax(Metric, axis=0)
    
    max_indices_j = np_argmax(Metric, axis=1)
    best_matches_4known = [(fin_list_graphs[j][0],known_complex_nodes_list[i],Metric[i,j],fin_list_graphs[j][1]) for i,j in enumerate(max_indices_j)]
    
    avged_f1_score4known = plot_f1_scores(best_matches_4known,out_comp_nm,'_best4known'+suffix,'Best predicted match for known complexes - ',plot_hist_flag)

    write_best_matches(best_matches_4known,out_comp_nm,dir_nm,'_best4known' + suffix,0,gt_names)

    if len(gt_names) == len(known_complex_nodes_list):
        best_matches_4predicted = [(fin_list_graphs[j][0],known_complex_nodes_list[i],Metric[i,j],fin_list_graphs[j][1],gt_names[i]) for j,i in enumerate(max_indices_i)]
    else:
        best_matches_4predicted = [(fin_list_graphs[j][0],known_complex_nodes_list[i],Metric[i,j],fin_list_graphs[j][1]) for j,i in enumerate(max_indices_i)]
        
    avged_f1_score4pred = plot_f1_scores(best_matches_4predicted,out_comp_nm,'_best4predicted'+suffix,'Best known match for predicted complexes - ',plot_hist_flag)
    
    avg_f1_score = (avged_f1_score4known + avged_f1_score4pred)/2
    net_f1_score = 2 * avged_f1_score4known * avged_f1_score4pred / (avged_f1_score4known + avged_f1_score4pred)
    
    write_best_matches_best4pred(best_matches_4predicted,out_comp_nm,dir_nm,'_best4predicted' + suffix)

    prec_FMM, recall_FMM, f1_FMM, max_matching_edges = f1_fmm(Metric)
    
    if plot_pr_flag:
        plot_pr_curve_FMM(Metric,fin_list_graphs,out_comp_nm+suffix)
    
    n_matches = int(len(max_matching_edges)/2)
    
    return avg_f1_score, net_f1_score,PPV,Sn,acc_unbiased,prec_FMM, recall_FMM, f1_FMM, n_matches


def f1_qi(Metric):
    '''
    Compute Qi et al metrics from similarity matrix as input

    Parameters
    ----------
    Metric : numpy.ndarray
        Matrix of match scores between test and predicted complexes

    Returns
    -------
    prec: float
        Qi et al Precision
    recall: float
        Qi et al Recall        

    '''
    max_i = Metric.max(axis=0)
    prec = sum(max_i)/len(max_i)
    
    max_j = Metric.max(axis=1)
    recall = sum(max_j)/len(max_j)    
    
    return prec,recall


def plot_pr_curve_orig(Metric,fin_list_graphs,out_comp_nm):
    '''
    Plot PR curve for Qi et measure using different Qi overlap measure thresholds

    Parameters
    ----------
    Metric : numpy.ndarray
        Matrix of match scores between test and predicted complexes
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores
    out_comp_nm : str
        Output directory name

    Returns
    -------
    None.

    '''
    
    n_divs = 10
    scores_list = [float(pred_complex[1]) for pred_complex in fin_list_graphs]
    #print(scores_list)
    min_score = min(scores_list)
    interval_len = (max(scores_list) - min_score)/float(n_divs)
    thresholds = [min_score + i*interval_len for i in range(n_divs)]
    
    precs = []
    recalls = []
    for thres in thresholds:
        # list of indices with scores greater than the threshold
        col_inds = [j for j,score in enumerate(scores_list) if score >= thres]
        
        prec, recall = f1_qi(Metric[:,col_inds])
        
        precs.append(prec)
        recalls.append(recall)
        
    fig = plt_figure()
    plt_plot(recalls,precs,'.-')
    plt_ylabel("Precision")
    plt_xlabel("Recall")
    plt_title("PR curve for Qi et al measure")
    plt_savefig(out_comp_nm + '_pr_qi.png')
    plt_close(fig)    
    
    
def node_comparison_prec_recall(known_complex_nodes_list, fin_list_graphs, N_pred_comp, N_test_comp, p, out_comp_nm,plot_pr_flag=False):
    '''
    Compute Qi et al metrics

    Parameters
    ----------
    known_complex_nodes_list : list[list[str]]
        List of list of nodes (proteins) in each known complex.
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores
    N_pred_comp : int
        No. of predicted complexes
    N_test_comp : int
        No. of test complexes
    p : float
        Threshold in Qi overlap measure
    out_comp_nm : str
        Output directory name
    plot_pr_flag : bool, optional
        Plot PR curve or not. The default is False.

    Returns
    -------
    Precision: float
        Qi et al Precision
    Recall: float
        Qi et al Recall
    F1_score: float
        Qi et al F1 score

    '''
    N_matches_test = 0

    Metric = np_zeros((N_test_comp, N_pred_comp))

    for i, test_complex in enumerate(known_complex_nodes_list):
        N_match_pred = 0
        for j, pred_complex in enumerate(fin_list_graphs):
            T = set(test_complex)
            P = pred_complex[0]
            C = len(T.intersection(P))
            A = len(P.difference(T))
            B = len(T.difference(P))
            
            if (A+C) == 0 or (B+C) == 0:
                return 0,0,0

            if float(C) / (A + C) > p and float(C) / (B + C) > p:
                Metric[i, j] = 1
                N_match_pred = N_match_pred + 1

        if N_match_pred > 0:
            N_matches_test = N_matches_test + 1
    if plot_pr_flag:        
        plot_pr_curve_orig(Metric,fin_list_graphs,out_comp_nm)

    Recall = float(N_matches_test) / N_test_comp

    N_matches_pred = np_count_nonzero(np_sum(Metric, axis=0))
    Precision = float(N_matches_pred) / N_pred_comp

    if Precision == Recall == 0:
        F1_score = 0
    else:
        F1_score = 2 * Precision * Recall / (Precision + Recall)

    return Precision, Recall, F1_score


def plot_size_dists(known_complex_nodes_list, fin_list_graphs, sizes_orig, out_comp_nm):
    '''
    Plot size (no. of nodes in a complex) distributions of known and predicted and complexes

    Parameters
    ----------
    known_complex_nodes_list : list[list[str]]
        List of list of nodes (proteins) in each known complex.
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores
    sizes_orig : list[int]
        List of no. of nodes in predicted complexes
    out_comp_nm : str
        Output directory name

    Returns
    -------
    None.

    '''
    sizes_known = [len(comp) for comp in known_complex_nodes_list]
    # Size distributions
    sizes_new = [len(comp[0]) for comp in fin_list_graphs]
    fig = plt_figure(figsize=(8,6),dpi=96)
    plt_rc('font', size=14) 

    if len(set(sizes_known)) <= 1:
        return
    sns_displot(sizes_known, kind='kde', label="known")
    if len(set(sizes_orig)) <= 1:
        return
    sns_displot(sizes_orig, kind='kde', label="predicted")
    if len(set(sizes_new)) <= 1:
        return
    sns_displot(sizes_new, kind='kde', label="predicted_known_prots")
    plt_ylabel("Probability density")    
    plt_xlabel("Complex Size (number of proteins)")
    plt_title("Complex size distributions")
    plt_savefig(out_comp_nm + '_size_dists_known_pred.png')
    plt_close(fig)


def remove_unknown_prots(fin_list_graphs_orig, prot_list, min_len = 3):
    '''
    Remove all proteins in predicted complexes that are not present in known complex protein list
    

    Parameters
    ----------
    fin_list_graphs_orig : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores
    prot_list : set(str)
        List of nodes (proteins) from known complexes. 
    min_len : int, optional
        Minimum no. of proteins that need to be there in a complex. The default is 3.

    Returns
    -------
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores

    '''
    fin_list_graphs = []
    for comp in fin_list_graphs_orig:
        comp = (comp[0].intersection(prot_list), comp[1])

        if len(comp[0]) > min_len - 1:  # Removing complexes with only one,two or no nodes
            fin_list_graphs.append(comp)
    return fin_list_graphs


def compute_metrics(known_complex_nodes_list, fin_list_graphs,out_comp_nm,N_test_comp,N_pred_comp,inputs,suffix,gt_names,plot_hist_flag=1):
    '''
    
    Compute evaluation metrics when predicted and known complexes are compared.

    Parameters
    ----------
    known_complex_nodes_list : list[list[str]]
        List of list of nodes (proteins) in each known complex.
    fin_list_graphs : list[(set(str),float)]
        List of tuples of set of predicted complex nodes and their scores
    out_comp_nm : str
        Output directory name
    N_test_comp : int
        No. of test complexes
    N_pred_comp : int
        No. of predicted complexes
    inputs : dict
        input parameters and their values
    suffix : str
        suffix for results: train, test or both 
    gt_names : TYPE(list[string]), optional
        List of cluster names for each cluster in gt_lines in the same orders. The default is [].
    plot_hist_flag : bool, optional
        Plot F1 score histograms of best matches or not. The default is 1.

    Returns
    -------
    eval_metrics_dict : dict
        Dictionary of evaluation metrics and their values

    '''
    eval_metrics_dict = dict()

    if N_test_comp != 0 and N_pred_comp != 0:
        Precision, Recall, F1_score = node_comparison_prec_recall(known_complex_nodes_list,fin_list_graphs, N_pred_comp, N_test_comp, inputs["eval_p"],out_comp_nm+suffix)
        
        avg_f1_score, net_f1_score,PPV,Sn,acc_unbiased,prec_FMM, recall_FMM, f1_FMM,n_matches = one2one_matches(known_complex_nodes_list, fin_list_graphs, N_pred_comp, N_test_comp,out_comp_nm,suffix,inputs['dir_nm'],0,gt_names,plot_hist_flag)
        
        with open(out_comp_nm + '_metrics.txt', "a") as fid:
            print("No. of matches in FMM = ", n_matches, file=fid)            
            print("FMM Precision = %.3f" % prec_FMM, file=fid)
            print("FMM Recall = %.3f" % recall_FMM, file=fid)
            print("FMM F1 score = %.3f" % f1_FMM, file=fid)               
            print("CMFF = %.3f" % net_f1_score, file=fid)   
            print("Unbiased PPV = %.3f" % PPV, file=fid)
            print("Unbiased Sn = %.3f" % Sn, file=fid)
            print("Unbiased accuracy= %.3f" % acc_unbiased, file=fid)             
            print("Net Averaged F1 score (Average of Precision and Recall based on F1 score) = %.3f" % avg_f1_score, file=fid)
            print("Prediction Precision = %.3f" % Precision, file=fid)
            print("Prediction Recall = %.3f" % Recall, file=fid)
            print("Prediction F1 score = %.3f" % F1_score, file=fid)    
            
        eval_metrics_dict = {"No. of matches in FMM": n_matches,"FMM Precision":prec_FMM,"FMM Recall":recall_FMM,"FMM F1 score":f1_FMM,"CMFF":net_f1_score,"Qi Precision": Precision,"Qi Recall":Recall,"Qi F1 score":F1_score}
    return eval_metrics_dict
    

def eval_complex(rf=0, rf_nm=0, inputs={}, known_complex_nodes_list=[], prot_list={}, fin_list_graphs=[],suffix="both"):
    '''
    Evaluate predicted complexes against known complexes
    
    Parameters
    ----------
    rf : bool, optional
        read flag to read complexes from file. The default is 0.
    rf_nm : str, optional
        Name of file to read results from. The default is 0 or ''.
    inputs : dict
        input parameters and their values. The default is {}.
    known_complex_nodes_list : list[list[str]], optional
        List of list of nodes (proteins) in each known complex. The default is [].       
    prot_list : set(str), optional
        List of nodes (proteins) from known complexes. The default is {}.
    fin_list_graphs : list[(set(str),float)], optional
        List of tuples of set of predicted complex nodes and their scores. The default is [].       
    suffix : str
        suffix for results: train, test or both. The default is "both".

    Returns
    -------
    None.

    '''
    logging_info("Evaluating complexes..." + suffix)
    out_comp_nm = inputs['dir_nm'] + inputs['out_comp_nm']
    

    if rf == 1:
        if rf_nm == 0:
            rf_nm = out_comp_nm + '_pred.txt'
        with open(rf_nm) as fn:
            fin_list_graphs = [(set(line.rstrip('\n').split()),1) for line in fn]  # Space separated text only
            # Just list of list of nodes

    sizes_orig = [len(comp[0]) for comp in fin_list_graphs]

    N_pred_comp = len(fin_list_graphs)
    if N_pred_comp == 0:
        return
    N_test_comp = len(known_complex_nodes_list)

    with open(out_comp_nm + '_metrics.txt', "a") as fid:
        print("No. of known complexes = ", N_test_comp, file=fid) 
        print("No. of predicted complexes = ", N_pred_comp, file=fid)       
        print("\n -- Metrics on complexes with all proteins -- ", file=fid)       
    
    compute_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm,N_test_comp,N_pred_comp,inputs,suffix+'_all_prots')            
    
    fin_list_graphs = remove_unknown_prots(fin_list_graphs, prot_list)
    plot_size_dists(known_complex_nodes_list, fin_list_graphs, sizes_orig, out_comp_nm)
    
    N_pred_comp = len(fin_list_graphs)
    with open(out_comp_nm + '_metrics.txt', "a") as fid:
        print("No. of predicted complexes after removing non-gold std proteins = ", N_pred_comp, file=fid)      
        print("\n -- Metrics on complexes with only gold std proteins -- ", file=fid)   
    
    compute_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm,N_test_comp,N_pred_comp,inputs,suffix+'_gold_std_prots')            
    with open(out_comp_nm + '_metrics.txt', "a") as fid:
        print("-- Finished writing main metrics -- \n", file=fid)   

    logging_info("Finished evaluating basic metrics for complexes " + suffix)