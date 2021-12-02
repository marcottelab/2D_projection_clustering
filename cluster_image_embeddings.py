# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:43:45 2021

@author: Meghana
"""

from eval_complex import compute_metrics
from sklearn.cluster import DBSCAN,AffinityPropagation,MeanShift,OPTICS,Birch, estimate_bandwidth
from sklearn.metrics import silhouette_score
from img2vec_pytorch import Img2Vec
from PIL import Image
from matplotlib import cm
import mrcfile
import numpy as np
import pandas as pd
import os


def get_config(dataset='real'):
    '''
    Get input and output config parameters for real or synthetic datasets
    
    Parameters:
    dataset (string): can be 'real' or 'synthetic'
    
    Returns:
    images_file_name (string): input mrcs file containing stack of images
    images_true_labels (string): true cluster labels in the format: name list_of_image_indices
    sep (string): Separater between name of cluster and its members
    index_start (integer): Index at which images are started being numbered, ex: 0, 1, etc
    out_dir (string): output directory name for results (will be created if it does not exist)

    '''
    if dataset == 'real':
        images_file_name = './data/real_dataset/mixture_2D.mrcs' 
        images_true_labels = './data/real_dataset/mixture_classification.txt'
        sep ='   '
        index_start = 1
        out_dir = 'results_real_dataset'
    else: # synthetic
        images_file_name = './data/synthetic_dataset/synthetic_2D.mrcs' 
        images_true_labels = './data/synthetic_dataset/synthetic_true_clustering.txt'
        sep = '\t'
        index_start = 0
        out_dir = 'results_synthetic_dataset'
        
    return images_file_name,images_true_labels,sep,index_start,out_dir


def read_clusters(images_true_labels,sep):
    '''
    Reads clustered labels

    Parameters:    
    images_true_labels (string): true cluster labels in the format: name list_of_image_indices
    sep (string): Separater between name of cluster and its members

    Returns:
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order
    
    '''    
    with open(images_true_labels) as f:
        raw_lines = f.readlines()
        list_of_tuples = [line.rstrip().split(sep) for line in raw_lines]
        gt_names = [entry[0] for entry in list_of_tuples]
        #gt_names = [line[0:sep] for line in raw_lines]
        #gt_lines = [set(line.rstrip()[sep:-1].split(', ')) for line in raw_lines]
        # Assuming format is [4,5,23,..]
        gt_lines = [set(line[1].lstrip()[1:-1].split(', ')) for line in list_of_tuples]
        
    return gt_lines, gt_names

    
def read_data(images_file_name, images_true_labels, sep):
    '''
    Reads MRC data and ground truth labels

    Parameters:    
    images_file_name (string): input mrcs file containing stack of images
    images_true_labels (string): filename of file with true cluster labels in the format: name list_of_image_indices
    sep (string): Separater between name of cluster and its members

    Returns:
    data (numpy.ndarray): Image stack of 2D numpy arrays
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order
    
    '''
    gt_lines, gt_names =  read_clusters(images_true_labels,sep)
    
    mrc = mrcfile.open(images_file_name, mode='r+')
    data = mrc.data
    
    print('Length of data= ', len(data))
    
    arr_tmp = np.array(data[0])
    print('Image array dimensions= ', arr_tmp.shape)
    
    mrc.close()
    
    # Remove unknown complex for evaluation
    for i, name in enumerate(gt_names):
        if name == 'unk':
            gt_names.pop(i)
            gt_lines.pop(i)

    return data, gt_lines, gt_names


def get_image_embedding(data,embedding_model = 'resnet-18'):  
    '''
    Get vector embeddings for each image in the data using a neural network model from img2vec
    
    Parameters:    
    data (numpy.ndarray): Image stack of 2D numpy arrays
    embedding_model (string): Model name - one of 'alexnet', 'vgg','densenet','resnet-18'
    
    Returns:
    vectors (numpy.ndarray): Array of image embeddings 
    '''
    # Initialize Img2Vec
    img2vec = Img2Vec(model=embedding_model) 
    
    data_nonneg = []
    for myarray in data:
        myarray[myarray <= 0] = 0
        data_nonneg.append(myarray)
        
    data = data_nonneg
    
    # Images are in circles in the below
    #list_of_PIL_imgs = [Image.fromarray(np.uint8(myarray*255),'L') for myarray in data] # grey scale doesnt work with image embeddings, but this looks same as other, so no prob
    list_of_PIL_imgs = [Image.fromarray(np.uint8(myarray*255)).convert('RGB') for myarray in data] # This gives a weird circle
    
    # This gives a black and white image, but atleast shape is visible well. Both L and RGB give same shape
    #list_of_PIL_imgs = [Image.fromarray(np.uint8(cm.gist_earth(myarray)*255)).convert('RGB') for myarray in data]
    #list_of_PIL_imgs = [Image.fromarray(np.uint8(cm.gist_earth(myarray)*255)).convert('L') for myarray in data]
    # Remove all negative values. You get same image if you do img[img < 0] = 0 # does'nt do anything, still negatives
    
    # The below two gives unintelligle figures
    #list_of_PIL_imgs = [Image.fromarray(myarray,'L') for myarray in data] 
    #list_of_PIL_imgs = [Image.fromarray(myarray,'RGB') for myarray in data] 
    
    vectors = img2vec.get_vec(list_of_PIL_imgs)
    
    return vectors


def get_cluster_wise_indices(n_clus,labels_clustered):
    '''
    Convert list of data index-wise cluster member names to clusterwise data indices
    
    Parameters:
    n_clus (int): No. of clusters predicted
    labels_clustered (list[int]): List of non-overlapping cluster numbers each data point belongs to. 
        List indices correspond to the data point in that index in the data matrix
        
    Returns:
    clusterwise_indices (list[list[int]]): List of clusters with members given by their indices

    '''
    start_ind = 0
    end_ind = n_clus
    if -1 in labels_clustered: # Only one cluster
        start_ind = -1
        end_ind = n_clus-1
        
    clusterwise_indices = [list(np.where(labels_clustered == i)[0]) for i in range(start_ind,end_ind)]
    return clusterwise_indices


def cluster_data(data_to_cluster,clustering_method,index_start):
    '''
    Cluster vectors of image embeddings 
    
    Parameters:
    data_to_cluster (numpy.ndarray): Array of image embeddings 
    clustering_method (sklearn.cluster._method.METHOD): method is a sckit learn clustering method with parameters, 
        ex: DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation(), etc     
    index_start (integer): Index at which images are started being numbered, ex: 0, 1, etc
    
    Returns:
    n_clus (int): No. of clusters predicted
    clusterwise_indices_str (list[tuple(set(string),float)]): List of clusters, each is a tuple of the cluster indices and cluster score (1 by default)
        ex: [({'0', '1'}, 1)]
    unsupervised_score_silhouette (float): Silhouette score for the clustering, range between [-1,1]. Higher score implies better separability of clusters,
        -2 if only one cluster or can't be otherwise computed
    '''
    
    if clustering_method == MeanShift():
        bandwidth = estimate_bandwidth(data_to_cluster)
        clustering_method = MeanShift(bandwidth = bandwidth)
        
    clustering = clustering_method.fit(data_to_cluster)
    labels_clustered = clustering.labels_
    print(clustering_method)
    
    try:
        unsupervised_score_silhouette = silhouette_score(data_to_cluster, labels_clustered, metric="sqeuclidean")
    except:
        unsupervised_score_silhouette = -2       
    n_clus = len(set(labels_clustered))
    print('No. of clusters = ',n_clus)

    clusterwise_indices = get_cluster_wise_indices(n_clus,labels_clustered)
    clusterwise_indices_str = [(set([str(ind+index_start) for ind in arr]),1) for arr in clusterwise_indices] 
    #print(clusterwise_indices_str)
    
    return n_clus, clusterwise_indices_str,unsupervised_score_silhouette


def evaluate_embeddings(vectors, gt_lines, index_start):
    '''
    Evaluates embeddings with true cluster labels using different distance measures to find best distance measure to use for clustering
    Parameters:    
    vectors (numpy.ndarray): Array of image embeddings     
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    index_start (integer): Index at which images are started being numbered, ex: 0, 1, etc
    
    Returns:
    silhouette_dict (dict): Dictionary of silhouette scores per distance measure    
    '''
    
    # Convert cluster wise labels to index wise cluster labels
    
    image_wise_cluster_labels = [-1]*len(vectors)
    
    for cluster_ind,cluster in enumerate(gt_lines):
        for image_ind in cluster:
            image_wise_cluster_labels[int(image_ind)-index_start] = cluster_ind
            
    distance_measures = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
                         'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 
                         'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule',
                         'l1', 'l2', 'manhattan']
    
    silhouette_dict = dict()
    best_silhouette = -1
    for distance in distance_measures:
        try:
            unsupervised_score_silhouette = silhouette_score(vectors, image_wise_cluster_labels, metric=distance)
        except:
            unsupervised_score_silhouette = -2
            
        silhouette_dict[distance] = unsupervised_score_silhouette
        
        if unsupervised_score_silhouette >= best_silhouette:
            best_silhouette = unsupervised_score_silhouette
            best_distance = distance
            
    silhouette_dict['max_silhouette_distance']= best_distance    
        
    return silhouette_dict        
            
    
def write_clusters(clusterwise_indices_start_str,clustering_method,out_dir):
    '''
    Write predicted clusters to file
    
    Parameters:    
    clusterwise_indices_start_str (list[tuple(set(string),float)]): List of clusters, each is a tuple of the cluster indices and cluster score (1 by default)
    clustering_method (sklearn.cluster._method.METHOD): method is a sckit learn clustering method with parameters, 
        ex: DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation(), etc   
    out_dir (string): output directory name for results (will be created if it does not exist)
    
    '''
    writeable_clusters = [" ".join(sorted(list(cluster[0]),key = lambda x: int(x)))+"\n" for cluster in clusterwise_indices_start_str]
    with open('./' + out_dir + '/' + str(clustering_method) + '_clusters_found.txt', "w") as fid:
        fid.writelines(writeable_clusters)    
        
        
def evaluate_clusters(clusterwise_indices_start_str,gt_lines,n_clus,clustering_method,out_dir,n_true_clus,gt_names):
    '''
    Evaluate predicted clusters against ground truth
    
    Parameters:    
    clusterwise_indices_start_str (list[tuple(set(string),float)]): List of clusters, each is a tuple of the cluster indices and cluster score (1 by default)
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    n_clus (int): No. of clusters predicted
    clustering_method (sklearn.cluster._method.METHOD): method is a sckit learn clustering method with parameters, 
        ex: DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation(), etc   
    out_dir (string): output directory name for results (will be created if it does not exist)
    n_true_clus (int): No. of true clusters
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order
    
    Returns:
    eval_metrics_dict (dict): Dictionary of evaluation metrics and their values on the predicted set of clusters w.r.t true clusters
    '''    

    out_dir = out_dir + '/evaluation_metrics'
    if not os.path.exists('./' + out_dir):
        os.mkdir('./' + out_dir)
    eval_metrics_dict = compute_metrics(gt_lines, clusterwise_indices_start_str,'./' + out_dir + '/' + str(clustering_method),len(gt_lines),n_clus,{"eval_p":0.5,"dir_nm":out_dir},'',gt_names)            
    with open('./' + out_dir + '/' + str(clustering_method) + '_metrics.txt', "a") as fid:
        print('No. of predicted clusters = ',n_clus, file=fid)  
        print('No. of true clusters = ',n_true_clus, file=fid)          
        
    eval_metrics_dict["No. of clusters"] = n_clus
        
    if ('MMR F1 score' in eval_metrics_dict) and ('Net F1 score' in eval_metrics_dict) and ('Qi F1 score' in eval_metrics_dict):
        eval_metrics_dict['3 F1 score average'] = (eval_metrics_dict['MMR F1 score'] + eval_metrics_dict['Net F1 score'] + eval_metrics_dict['Qi F1 score'])/3.0

    return eval_metrics_dict


def evaluate_SLICEM(gt_lines,gt_names,n_true_clus,dataset,sep,index_start):
    '''
    Evaluate SLICEM clustering on synthetic dataset
    
    Parameters:    
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    gt_names (list[string]): List of cluster names for each cluster in gt_lines in the same order    
    n_true_clus (int): No. of true clusters
    
    Returns:
    eval_metrics_dict (dict): Dictionary of evaluation metrics and their values on the predicted set of clusters w.r.t true clusters
    
    '''
    if dataset == 'synthetic':
        out_dir = 'data/synthetic_dataset'        
    else: # real 
        out_dir = 'data/real_dataset'     

    SLICEM_labels_file =  './' + out_dir + '/slicem_clustering.txt'

    cluster_lines, cluster_numbers =  read_clusters(SLICEM_labels_file,sep)
    n_clus = len(cluster_lines)
    
    cluster_lines = [set([str(int(img_ind) + index_start) for img_ind in entry]) for entry in cluster_lines]
    # Adding score as 1 to satisfy input format for compute_metrics function
    clusterwise_indices_start_str = [(entry,1) for entry in cluster_lines]
    
    eval_metrics_dict = evaluate_clusters(clusterwise_indices_start_str,gt_lines,n_clus,'SLICEM',out_dir,n_true_clus,gt_names)
    eval_metrics_dict['Silhouette score'] = 'NA'
    return eval_metrics_dict


def main():
# Main driver
    embedding_methods = ['alexnet', 'vgg','densenet','resnet-18']
    clustering_methods = [DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation()]
    datasets = ['real','synthetic']
    #datasets = ['real']
    #datasets = ['synthetic']
    
    for dataset in datasets:
        images_file_name,images_true_labels,sep,index_start,out_dir_orig = get_config(dataset)
        
        if not os.path.exists('./' + out_dir_orig):
            os.mkdir('./' + out_dir_orig)
            
        data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep)
        n_true_clusters = len(gt_lines)
        
        #data_to_cluster = data.flatten().reshape(100,96*96)  # Just flattened data: Check if correct
        
        results_df = pd.DataFrame()
        embedding_eval_df = pd.DataFrame()
        
        for embedding_method in embedding_methods:
            out_dir_emb = out_dir_orig + '/'+embedding_method
            if not os.path.exists('./' + out_dir_emb):
                os.mkdir('./' + out_dir_emb)
            data_to_cluster = get_image_embedding(data,embedding_method)
            
            silhouette_dict = evaluate_embeddings(data_to_cluster, gt_lines, index_start)
            if len(embedding_eval_df) == 0:
                embedding_eval_df = pd.DataFrame(columns = silhouette_dict.keys())    
                
            embedding_eval_df = embedding_eval_df.append(pd.Series(silhouette_dict,name = embedding_method))
            
            # try PCA before clustering . Set distance threshold based on P c a variance 
            
            for clustering_method in clustering_methods:
                out_dir = out_dir_emb + '/'+str(clustering_method)  
                if not os.path.exists('./' + out_dir):
                    os.mkdir('./' + out_dir)
                                 
                n_clus, clusterwise_indices_str,unsupervised_score_silhouette = cluster_data(data_to_cluster,clustering_method,index_start)
                write_clusters(clusterwise_indices_str,clustering_method,out_dir)                
                eval_metrics_dict = evaluate_clusters(clusterwise_indices_str,gt_lines,n_clus,embedding_method + '_' + str(clustering_method),out_dir,n_true_clusters,gt_names)
                eval_metrics_dict['Silhouette score'] = unsupervised_score_silhouette
                if len(results_df) == 0:
                    results_df = pd.DataFrame(columns = eval_metrics_dict.keys())
                results_df = results_df.append(pd.Series(eval_metrics_dict,name = embedding_method + ' embedding ' +  str(clustering_method) + ' clustering'))
        
        eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clusters,dataset,sep,index_start)
        results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = 'SLICEM'))
            
        #results_df.sort_values(by='No. of clusters',key=lambda x: abs(x-n_true_clusters),inplace=True)
        #results_df.sort_values(by='3 F1 score average',ascending=False,inplace=True)
        if ('MMR F1 score' in eval_metrics_dict):
            results_df.sort_values(by='MMR F1 score',ascending=False,inplace=True)
            
        # check results stability 
        
        embedding_eval_df.to_csv('./' + out_dir_orig + '/evaluating_embeddings_' + dataset + '.csv')
        
        results_df.to_csv('./' + out_dir_orig + '/compiled_results_all_methods_sorted_' + dataset + '.csv')

        
if __name__ == "__main__":
    main()