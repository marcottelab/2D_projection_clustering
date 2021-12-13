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
from sklearn.decomposition import PCA, TruncatedSVD
from loguru import logger
from sklearn import manifold
from siamese_embedding import siamese_embedding
import mrcfile
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random


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
    
    logger.info("Length of data= {}", len(data))
    
    arr_tmp = np.array(data[0])
    logger.info('Image array dimensions= {}', arr_tmp.shape)
    
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
    
    # Initialize Img2Vec
    if embedding_model in ['alexnet', 'vgg','densenet','resnet-18']:
        img2vec = Img2Vec(model=embedding_model)     
        vectors = img2vec.get_vec(list_of_PIL_imgs)
    elif embedding_model == 'siamese':
        vectors = siamese_embedding(list_of_PIL_imgs)
    else:
        logger.error('Embedding model not found. Returning flattened images')
        vectors = data.flatten().reshape(100,96*96)  # Just flattened data: Check if correct

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


def cluster_data(data_to_cluster,clustering_method,index_start, dist_metric = ''):
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
    
    # Setting parameters
    if clustering_method == MeanShift():
        bandwidth = estimate_bandwidth(data_to_cluster)
        clustering_method = MeanShift(bandwidth = bandwidth)
        
    if len(dist_metric):
        logger.info("Setting best distance measure - {}", dist_metric)
        if clustering_method == DBSCAN():
            clustering_method = DBSCAN(metric = dist_metric)
        elif clustering_method == OPTICS():
            clustering_method = OPTICS(metric = dist_metric)
    else:
        dist_metric = "sqeuclidean"
        
    # Clustering
    clustering = clustering_method.fit(data_to_cluster)
    labels_clustered = clustering.labels_
    logger.info(clustering_method)
    
    try:
        unsupervised_score_silhouette = silhouette_score(data_to_cluster, labels_clustered, metric=dist_metric)
    except:
        unsupervised_score_silhouette = -2       
    n_clus = len(set(labels_clustered))
    logger.info('No. of clusters = {}', n_clus)

    clusterwise_indices = get_cluster_wise_indices(n_clus,labels_clustered)
    clusterwise_indices_str = [(set([str(ind+index_start) for ind in arr]),1) for arr in clusterwise_indices] 
    #print(clusterwise_indices_str)
    
    return n_clus, clusterwise_indices_str,unsupervised_score_silhouette


def get_image_wise_cluster_labels(vectors,gt_lines,index_start):
    '''
    Convert cluster wise labels to index wise cluster labels
    Parameters:    
    vectors (numpy.ndarray): Array of image embeddings     
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster
    index_start (integer): Index at which images are started being numbered, ex: 0, 1, etc

    Returns:
    image_wise_cluster_labels (list[int]): List of cluster numbers per image
    '''
    
    image_wise_cluster_labels = [-1]*len(vectors)
    
    for cluster_ind,cluster in enumerate(gt_lines):
        for image_ind in cluster:
            image_wise_cluster_labels[int(image_ind)-index_start] = cluster_ind
            
    return image_wise_cluster_labels


def myfun():
    '''
    Returns a constant number, useful to set constant random state
    '''
    return 0.42

def get_train_images(image_wise_cluster_labels,train_cluster_inds,vectors):
    '''
    Splits images into train and test sets based on train and test clusters, and provides corresponding cluster labels for each image in the train and teset set
    Parameters:
    image_wise_cluster_labels (list[int]): List of cluster numbers per image        
    train_cluster_inds (list[int]): List of indices of original ground truth list corresponding to train complexes
    vectors (numpy.ndarray): Array of image embeddings     
        
    Returns:
    train_image_wise_cluster_labels (list[int]): List of cluster numbers per train image  
    train_vectors (list[numpy.array]): Train image embeddings
    test_image_wise_cluster_labels (list[int]): List of cluster numbers per test image  
    test_vectors (list[numpy.array]): Test image embeddings
    
    '''
    train_image_wise_cluster_labels = []
    test_image_wise_cluster_labels = []
    train_vectors = []
    test_vectors = []
    
    for image_ind,cluster_ind in enumerate(image_wise_cluster_labels):
        if cluster_ind in train_cluster_inds:
            train_image_wise_cluster_labels.append(cluster_ind)
            train_vectors.append(vectors[image_ind])
        else:
            test_image_wise_cluster_labels.append(cluster_ind)
            test_vectors.append(vectors[image_ind])
            
    return train_image_wise_cluster_labels, train_vectors, test_image_wise_cluster_labels, test_vectors
            
            
def train_test_split_complexes(gt_lines,train_percentage=0.7):
    '''
    Splits complexes into train and test sets
    Parameters:        
    gt_lines (list[set(string)]): List of sets of image indices in string format per ground truth cluster    
    train_percentage (int): % of complexes to be in train set
    
    Returns:
    train_cluster_inds (list[int]): List of indices of original ground truth list corresponding to train complexes
    test_cluster_inds (list[int]): List of indices of original ground truth list corresponding to test complexes
    
    train_clusters (list[set(string)]): List of sets of image indices in string format per ground truth training cluster 
    test_clusters (list[set(string)]): List of sets of image indices in string format per ground truth test cluster 
    
    '''
    n_true_clus = len(gt_lines)
    inds = list(range(0,n_true_clus)) # Note: Indices start from 0 here
    random.shuffle(inds, myfun)
    
    train_last_ind = int(train_percentage*n_true_clus)
    
    train_cluster_inds = inds[:train_last_ind]
    test_cluster_inds = inds[train_last_ind+1:]
    
    train_clusters = []
    test_clusters = []
    for i, cluster in enumerate(gt_lines):
        if i in train_cluster_inds:
            train_clusters.append(cluster)
        else:
            test_clusters.append(cluster)
    
    return train_cluster_inds, test_cluster_inds, train_clusters, test_clusters
    
            
def evaluate_embeddings(vectors, image_wise_cluster_labels):
    '''
    Evaluates embeddings with true cluster labels using different distance measures to find best distance measure to use for clustering
    Parameters:    
    vectors (numpy.ndarray): Array of image embeddings     
    image_wise_cluster_labels (list[int]): List of cluster numbers per image
    
    Returns:
    silhouette_dict (dict): Dictionary of silhouette scores per distance measure    
    '''
            
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


def reduce_dimensions(vectors, n_dims = 50):
    '''
    Reduce dimensions with PCA or truncated SVD (for sparse embeddings)
    Parameters:    
    vectors (numpy.ndarray): Array of image embeddings 

    Returns:
    vectors_reduced (numpy.ndarray): Reduced dimensional Array of image embeddings   
    '''
    
    # Calculate sparsity
    logger.info('Total elements in vectors = {}', float(vectors.size))
    sparsity = 1.0 - ( np.count_nonzero(vectors) / float(vectors.size) )
    logger.info('Sparsity of original embeddings = {}', sparsity)
        
    if sparsity > 0.5:
        method = TruncatedSVD(n_components=n_dims)       
    else:
        method = PCA(n_components=n_dims)
        
    vectors_reduced = method.fit_transform(vectors) 
    variance_captured = sum(method.explained_variance_ratio_)         
        
    logger.info('Reduced dimensions: {}', vectors_reduced[0].shape)
    logger.info('Variance captured: {}', variance_captured)
    
    return vectors_reduced
    

def plot_tsne(vectors_reduced,out_dir_emb,image_wise_true_labels, dist_metric = 'euclidean'):
    '''
    Plots TSNE on all data using the specified metric
    Parameters:    
    vectors_reduced (numpy.ndarray): Reduced dimensional Array of image embeddings   
    image_wise_true_labels (list[int]): List of cluster numbers per image
    dist_metric (string): Pairwise distance metric

    '''
    method = manifold.TSNE(n_components=2, init="pca", random_state=0, metric = dist_metric)
    
    Y = method.fit_transform(vectors_reduced)
    plt.figure(figsize=(15, 8))
    plt.scatter(Y[:, 0], Y[:, 1], c = image_wise_true_labels, cmap='viridis')
    plt.axis("tight")
    plt.savefig('./' + out_dir_emb + '/embedding_tsne.jpg')
    
    
def main():
# Main driver
    embedding_methods = ['siamese']
    #embedding_methods = ['alexnet','densenet','resnet-18', 'vgg']
    clustering_methods = [DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation()]
    #datasets = ['real','synthetic']
    #datasets = ['real']
    datasets = ['synthetic']  
    
    for dataset in datasets:
        images_file_name,images_true_labels,sep,index_start,out_dir_orig = get_config(dataset)
        
        if not os.path.exists('./' + out_dir_orig):
            os.mkdir('./' + out_dir_orig)
            
        # Setting logger
        logger.add('./' + out_dir_orig + '/log_file.txt',level="INFO")
            
        data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep)
        n_true_clusters = len(gt_lines)
                
        results_df = pd.DataFrame()
        embedding_eval_df = pd.DataFrame()
        
        for embedding_method in embedding_methods:
            logger.info(embedding_method)
            out_dir_emb = out_dir_orig + '/'+embedding_method
            if not os.path.exists('./' + out_dir_emb):
                os.mkdir('./' + out_dir_emb)
            data_to_cluster = get_image_embedding(data,embedding_method)
            
            data_to_cluster = reduce_dimensions(data_to_cluster)
            
            image_wise_cluster_labels = get_image_wise_cluster_labels(data_to_cluster,gt_lines,index_start)
            
            # Split into train and test sets
            train_cluster_inds, test_cluster_inds, train_clusters, test_clusters = train_test_split_complexes(gt_lines)
            train_image_wise_cluster_labels, train_vectors, test_image_wise_cluster_labels, test_vectors = get_train_images(image_wise_cluster_labels,train_cluster_inds,data_to_cluster)
            
            silhouette_dict_full = evaluate_embeddings(data_to_cluster, image_wise_cluster_labels)
            
            silhouette_dict = evaluate_embeddings(train_vectors, train_image_wise_cluster_labels)
            
            if len(embedding_eval_df) == 0:
                embedding_eval_df = pd.DataFrame(columns = silhouette_dict.keys())    
                
            embedding_eval_df = embedding_eval_df.append(pd.Series(silhouette_dict,name = embedding_method + ' with train data'))
            embedding_eval_df = embedding_eval_df.append(pd.Series(silhouette_dict_full,name = embedding_method + ' with full data'))

            # Plot TSNEs
            try:
                plot_tsne(data_to_cluster,out_dir_emb,image_wise_cluster_labels, silhouette_dict['max_silhouette_distance'] )                
            except: # Add traceback
                logger.error('ERROR in tsne plot')
            
            # Set distance threshold based on PCA variance 
            
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