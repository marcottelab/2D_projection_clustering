# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:43:45 2021

@author: Meghana
"""
import traceback
import faulthandler
faulthandler.enable()

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import pickle as pkl

from argparse import ArgumentParser as argparse_ArgumentParser
from sklearn.cluster import DBSCAN,AffinityPropagation,OPTICS,Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.decomposition import PCA, TruncatedSVD
from loguru import logger
from sklearn import manifold
from yaml import dump as yaml_dump
from util.input_functions import get_config, read_data
from util.evaluation_functions import evaluate_SLICEM, evaluate_clusters
from siamese_embedding import siamese_embedding
from read_node_embeddings import slicem_graph_embeddings

#from clusteval import clusteval
#from matplotlib import cm


def get_image_embedding(data,embedding_model = 'resnet-18',combine_graph_flag=0,graph_embedding_method='',dataset='real',graph_name='slicem_edge_list',graph_type='directed',node_attribute_method='siamese'):  
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
    if embedding_model in ['alexnet', 'vgg','densenet','resnet-18','efficientnet_b1','efficientnet_b7']:
        img2vec = Img2Vec(model=embedding_model)     
        vectors = img2vec.get_vec(list_of_PIL_imgs)
        
        if combine_graph_flag:
            graph_vectors = slicem_graph_embeddings(dataset,graph_embedding_method,'stellar',graph_name,'',graph_type)
            
            vectors = np.hstack((vectors,graph_vectors))
            logger.info('Stacked image + graph embedding array shape: {}',np.shape(vectors))
            
    elif embedding_model in ['siamese','siamese_noisy','siamese_more_projs_all','siamese_more_projs_noisy','siamese_real','siamese_real_synthetic','siamese_more_projs_wo_4v6c','siamese_more_negs']:
        if embedding_model in ['siamese_more_projs_all','siamese_more_projs_noisy','siamese_real_synthetic','siamese_more_projs_wo_4v6c']:
            correct_dims = (100, 100)
        elif embedding_model == 'siamese_real':
            correct_dims = (96, 96)
        else:
            correct_dims = (350, 350)
        #if dataset == 'real': # Resize to correct input dimensions
        if dataset == 'real' or dataset == 'synthetic' or dataset == 'synthetic_noisy': # Resize to correct input dimensions        
            list_of_PIL_imgs = [im1.resize(correct_dims) for im1 in list_of_PIL_imgs]
        
        vectors = siamese_embedding(list_of_PIL_imgs, embedding_model)
        if combine_graph_flag:
            graph_vectors = slicem_graph_embeddings(dataset,graph_embedding_method,'stellar',graph_name,'',graph_type)
            vectors = np.hstack((vectors,graph_vectors))
    elif embedding_model in ['metapath2vec','wys','graphWave','node2vec']:
        vectors = slicem_graph_embeddings(dataset,embedding_model,'stellar',graph_name,'',graph_type)
    elif embedding_model == 'slicem-graph-' + str(graph_embedding_method):
        vectors = slicem_graph_embeddings(dataset,graph_embedding_method,'stellar',graph_name,'',graph_type)        
    elif embedding_model in ['graphSage','attri2vec','gcn','cluster_gcn','gat','APPNP']:
        vectors = slicem_graph_embeddings(dataset,embedding_model,'stellar',graph_name,node_attribute_method,graph_type)
    else:
        logger.error('Embedding model not found. Returning flattened images')
        vectors = data.flatten().reshape(100,96*96)  # Just flattened data: Check if correct
    logger.info('Embedding array shape: {}',np.shape(vectors))

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


def cluster_data(data_to_cluster,clustering_method,index_start, dist_metric = '',image_labels = None):
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
    unsupervised_score_calinski_harabasz (float): Calinski-Harabasz index also known as the Variance Ratio Criterion - can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters
        -2 if only one cluster or can't be otherwise computed
    unsupervised_score_davies_bouldin (float): Davies-Bouldin index - Zero is the lowest possible score. Values closer to zero indicate a better partition, 
        -2 if only one cluster or can't be otherwise computed
    '''
    
    # Setting parameters
        
    if len(dist_metric):
        logger.info("Setting best distance measure - {}", dist_metric)
        if clustering_method == DBSCAN():
            clustering_method = DBSCAN(metric = dist_metric)

            if dist_metric == "seuclidean":
                try:
                    clustering_method = DBSCAN(metric = dist_metric, metric_params={"V":1},algorithm='brute')
                except:
                    logger.error(traceback.format_exc())
        elif clustering_method == OPTICS():
            clustering_method = OPTICS(metric = dist_metric)
    # else:
    #     dist_metric = "sqeuclidean"
        
    # Clustering
    try:
        clustering = clustering_method.fit(data_to_cluster)
    except:
        logger.error('Error in clustering_method.fit')
        logger.error(traceback.format_exc())
        logger.error(clustering_method)
        logger.error(type(clustering_method))        
        return 0, '',-2, -2, -2
    labels_clustered = clustering.labels_
    logger.info(clustering_method)
    
    try:
        unsupervised_score_silhouette = silhouette_score(data_to_cluster, labels_clustered, metric=dist_metric)
    except:
        logger.error(traceback.format_exc())
        unsupervised_score_silhouette = -2 
        
    try:
        unsupervised_score_calinski_harabasz = calinski_harabasz_score(data_to_cluster, labels_clustered)
    except:
        unsupervised_score_calinski_harabasz = -2 
        logger.error(traceback.format_exc())
        
    try:
        unsupervised_score_davies_bouldin = davies_bouldin_score(data_to_cluster, labels_clustered)
    except:
        unsupervised_score_davies_bouldin = -2         
        logger.error(traceback.format_exc())
        
    n_clus = len(set(labels_clustered))
    logger.info('No. of clusters = {}', n_clus)

    clusterwise_indices = get_cluster_wise_indices(n_clus,labels_clustered)
    if not image_labels:
        clusterwise_indices_str = [(set([str(ind+index_start) for ind in arr]),1) for arr in clusterwise_indices] 
    else:
        clusterwise_indices_str = [(set([image_labels[ind] for ind in arr]),1) for arr in clusterwise_indices] 
        
     #print(clusterwise_indices_str)
    
    return n_clus, clusterwise_indices_str,unsupervised_score_silhouette, unsupervised_score_calinski_harabasz, unsupervised_score_davies_bouldin


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


def get_train_images(image_wise_cluster_labels,train_cluster_inds,vectors,index_start):
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
    train_images_orig_names = []
    test_images_orig_names = []
    
    train_vectors = []
    test_vectors = []
    
    for image_ind,cluster_ind in enumerate(image_wise_cluster_labels):
        if cluster_ind in train_cluster_inds:
            train_image_wise_cluster_labels.append(cluster_ind)
            train_vectors.append(vectors[image_ind])
            train_images_orig_names.append(str(image_ind+index_start))
        else:
            test_image_wise_cluster_labels.append(cluster_ind)
            test_vectors.append(vectors[image_ind])
            test_images_orig_names.append(str(image_ind))
            
    return train_image_wise_cluster_labels, train_vectors, test_image_wise_cluster_labels, test_vectors, train_images_orig_names, test_images_orig_names
             
            
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
                         'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    
    silhouette_dict = dict()
    best_silhouette = -1
    best_distance = 'euclidean' # default
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
            
    
def write_clusters(clusterwise_indices_start_str,clustering_method,out_dir,main_results_dir='../results'):
    '''
    Write predicted clusters to file
    
    Parameters:    
    clusterwise_indices_start_str (list[tuple(set(string),float)]): List of clusters, each is a tuple of the cluster indices and cluster score (1 by default)
    clustering_method (sklearn.cluster._method.METHOD): method is a sckit learn clustering method with parameters, 
        ex: DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation(), etc   
    out_dir (string): output directory name for results (will be created if it does not exist)
    
    '''
    writeable_clusters = [" ".join(sorted(list(cluster[0]),key = lambda x: int(x)))+"\n" for cluster in clusterwise_indices_start_str]
    with open(main_results_dir + '/' + out_dir + '/' + str(clustering_method) + '_clusters_found.txt', "w") as fid:
        fid.writelines(writeable_clusters)    


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
        logger.info('Performing Truncated SVD...')
        method = TruncatedSVD(n_components=n_dims)       
    else:
        logger.info('Performing PCA...')        
        method = PCA(n_components=n_dims)
        
    method.fit(vectors)        
    variance_per_component = method.explained_variance_ratio_
    logger.info('Variance per component {}', variance_per_component)
    
    # Remove dimensions contributing lesser than 2% variance
    
    min_contribution = 0.02
    n_imp_dims = 0
    for var in variance_per_component:
        if var > min_contribution:
            n_imp_dims += 1
        else:
            break
        
    if sparsity > 0.5:
        logger.info('Performing Truncated SVD...')
        method = TruncatedSVD(n_components=n_imp_dims)       
    else:
        logger.info('Performing PCA...')        
        method = PCA(n_components=n_imp_dims)    
    
    vectors_reduced = method.fit_transform(vectors)     
    variance_per_component = method.explained_variance_ratio_    
    variance_captured = sum(variance_per_component)           
      
    logger.info('Reduced dimensions: {}', vectors_reduced[0].shape)
    logger.info('Variance captured: {}', variance_captured)
    
    return vectors_reduced
    

def plot_tsne(vectors_reduced,out_dir_emb,image_wise_true_labels, gt_names, dist_metric = 'euclidean',main_results_dir='../results'):
    '''
    Plots TSNE on all data using the specified metric
    Parameters:    
    vectors_reduced (numpy.ndarray): Reduced dimensional Array of image embeddings   
    image_wise_true_labels (list[int]): List of cluster numbers per image
    dist_metric (string): Pairwise distance metric

    '''
    method = manifold.TSNE(n_components=2, init="pca", random_state=0, metric = dist_metric)
    
    Y = method.fit_transform(vectors_reduced)
    
    n_classes = len(set(image_wise_true_labels))
    
    diff_colors_list_fixed = [np.array([[0.3457559 , 0.55905827, 0.82117878]]), 
                              np.array([[0.26160068, 0.6841755 , 0.75952533]]), 
                              np.array([[0.38219951, 0.70277622, 0.28288882]]), 
                              np.array([[0.90228767, 0.59949946, 0.45473147]]), 
                              np.array([[0.46020073, 0.23543775, 0.48584243]]), 
                              np.array([[0.7827978 , 0.89998952, 0.98915007]]), 
                              np.array([[0.8462942 , 0.01103685, 0.77265595]]), 
                              np.array([[0.18074287, 0.42777851, 0.69501253]]), 
                              np.array([[0.9680689 , 0.83262389, 0.84637277]]), 
                              np.array([[0.53541501, 0.1221299 , 0.98310477]]), 
                              np.array([[0.5853483 , 0.88570956, 0.62921587]]), 
                              np.array([[0.72692145, 0.77815712, 0.86430729]]), 
                              np.array([[0.75677848, 0.81589046, 0.4499639 ]]), 
                              np.array([[0.72400129, 0.44154602, 0.72959349]]), 
                              np.array([[0.95657965, 0.97237277, 0.37744636]]), 
                              np.array([[0.26646396, 0.59516464, 0.57971231]]), 
                              np.array([[0.76000252, 0.21777765, 0.65589281]]), 
                              np.array([[0.79556958, 0.75577962, 0.3899597 ]]), 
                              np.array([[0.68897906, 0.80750367, 0.79761243]]), 
                              np.array([[0.89520731, 0.10052996, 0.65629483]]), 
                              np.array([[0.65480282, 0.94389686, 0.27497054]]), 
                              np.array([[0.43711918, 0.95591558, 0.69391039]]), 
                              np.array([[0.04287057, 0.89577048, 0.14567296]]), 
                              np.array([[0.00502088, 0.23801188, 0.0114243 ]]), 
                              np.array([[0.83189672, 0.97171536, 0.68429438]]), 
                              np.array([[0.26305008, 0.46343431, 0.31870855]]), 
                              np.array([[0.22394731, 0.503383  , 0.31687779]]), 
                              np.array([[0.63161589, 0.6717516 , 0.37782211]]), 
                              np.array([[0.56370655, 0.32879629, 0.03594461]]), 
                              np.array([[0.87672986, 0.06079454, 0.38370234]]), 
                              np.array([[0.81261996, 0.40369241, 0.87148381]]), 
                              np.array([[0.23532809, 0.95052347, 0.61813889]]), 
                              np.array([[0.65689502, 0.83417387, 0.94463902]]), 
                              np.array([[0.20394688, 0.49266594, 0.41829996]]), 
                              np.array([[0.62408518, 0.22560799, 0.4332037 ]])]

    if n_classes <= len(diff_colors_list_fixed):
        diff_colors_list = diff_colors_list_fixed
    else:    
        diff_colors_list = [np.array([[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]]) for i in range(n_classes)]

    markers = ["s","o",".","v","^","<",">","1","2","3","4","8","p","P","h","H","+","x","X","D","d"]
    fig, ax = plt.subplots()
    
    if -1 in image_wise_true_labels:
        gt_names.append('unk')
        
    for i, c in enumerate(np.unique(image_wise_true_labels)):
        if i < len(markers):
            mk = markers[i]
        else:
            mk = markers[2]
        ax.scatter(Y[:,0][image_wise_true_labels==c],Y[:,1][image_wise_true_labels==c],c=diff_colors_list[i], marker=mk, label = gt_names[c])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

    plt.savefig(main_results_dir + '/' + out_dir_emb + '/embedding_tsne.jpg', bbox_inches='tight', dpi=300)
    

# def cluster_hyperparameter_optimization_clusteval(data_to_cluster,clustering_method):
#     ce = clusteval(cluster='dbscan')

#     # Fit to find optimal number of clusters using dbscan
#     results= ce.fit(data_to_cluster)
    
#     # Make plot of the cluster evaluation
#     ce.plot()
    
#     # Make scatter plot. Note that the first two coordinates are used for plotting.
#     ce.scatter(data_to_cluster)
    
#     # results is a dict with various output statistics. One of them are the labels.
#     cluster_labels = results['labx']
    
#     print(cluster_labels)
#     # Giving error with silhouette score not defined when each individual is a cluster
    
    
def make_generator(parameters):
    '''
    Generator for parameters

    Parameters
    ----------
    parameters : dict(string: list[float])
        A dictionary of parameters and their range of values

    Yields
    ------
    dict
        A dictionary of parameter and its value

    '''
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {p : parameters[p]
                    for p in parameters if p != key_to_iterate}
        for val in parameters[key_to_iterate]:
            for pars in make_generator(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res
        
        
def cluster_hyperparameter_optimization(cluster_hyper_param_ranges,data_to_cluster,image_wise_cluster_labels,index_start,embedding_method,gt_lines,gt_names,n_true_clusters,out_dir_orig,dataset,dist_metric='',main_results_dir='../results',graph_embedding_method = '',node_attribute_method=''):
    '''
    
    Find best clustering algorithm and hyperparameters using the provided methods and ranges, evaluated on the training set

    Parameters
    ----------
    cluster_hyper_param_ranges : TYPE (dict(str: dict(str: iterable)))
        DESCRIPTION. Dictionary of clustering method, each a dictionary of parameter name with an iterable of values 
    data_to_cluster : TYPE (numpy.ndarray)
        DESCRIPTION.: Array of image embeddings
    image_wise_cluster_labels : TYPE (list[int])
        DESCRIPTION.: List of cluster numbers per image
    index_start : TYPE(integer)
        DESCRIPTION.: Index at which images are started being numbered, ex: 0, 1, etc
    embedding_method : TYPE (string)
        DESCRIPTION. Name of embedding method
    gt_lines : TYPE (list[set(string)])
        DESCRIPTION.: List of sets of image indices in string format per ground truth cluster
    gt_names : TYPE(list[string])
        DESCRIPTION.: List of cluster names for each cluster in gt_lines in the same order
    n_true_clusters : TYPE (int)
        DESCRIPTION. No. of true clusters
    out_dir_orig : TYPE (str)
        DESCRIPTION. Name of the output files' directory
    dataset : TYPE (str)
        DESCRIPTION. Name of the dataset
    dist_metric : TYPE, optional (string)
        DESCRIPTION. The default is ''.: Pairwise distance metric

    Returns
    -------
    best_method_sklearn: TYPE (str)
        DESCRIPTION. Name of the best method details
    best_method: TYPE (str)
        DESCRIPTION. Name of the best method    

    '''
    results_df = pd.DataFrame()
    for method, param_grid in cluster_hyper_param_ranges.items():
        logger.debug(list(make_generator(param_grid)))
        for params in make_generator(param_grid):
                
            if method == "DBSCAN":
                if len(dist_metric):
                    logger.info("Setting best distance measure - {}", dist_metric)
                    fixed_params = {"metric": dist_metric}
                    if dist_metric == "seuclidean":
                        fixed_params.update({"metric_params":{"V":1},"algorithm":'brute'})
                    params.update(fixed_params)
                ca = DBSCAN( **params )
            if method == "OPTICS":
                if len(dist_metric):
                    logger.info("Setting best distance measure - {}", dist_metric)
                    fixed_params = {"metric": dist_metric}
                    params.update(fixed_params)
                ca = OPTICS( **params )
            if method == "Birch":
                ca = Birch( **params )
            if method == "AffinityPropagation":
                ca = AffinityPropagation( **params )
                
            logger.info(ca)
            out_dir_emb = out_dir_orig + '/'+embedding_method + str(graph_embedding_method)+node_attribute_method
            
            cluster_method_str = method + '_'  
            for param,value in params.items():
                try:
                    value = round(value,2)
                except:
                    value = value
                if (param != "metric") and (param != "metric_params") and (param != "algorithm"):
                    cluster_method_str = cluster_method_str + param + '_' + str(value)
                    
            out_dir = out_dir_emb + '/tuning'  
            if not os.path.exists(main_results_dir + '/' + out_dir):
                os.mkdir(main_results_dir + '/' + out_dir)
            out_dir = out_dir + '/'+cluster_method_str  
            if not os.path.exists(main_results_dir + '/' + out_dir):
                os.mkdir(main_results_dir + '/' + out_dir) 
            
            warm_start = 0
            if os.path.exists(main_results_dir + '/' + out_dir + '/eval_metrics_dict.pkl'):
                warm_start = 1
                
            if warm_start:
                with open(main_results_dir + '/' + out_dir + '/eval_metrics_dict.pkl','rb') as f:
                    eval_metrics_dict = pkl.load(f)   
            else:
                try:
                    logger.info("Clustering...")
                    n_clus, clusterwise_indices_str,unsupervised_score_silhouette,unsupervised_score_calinski_harabasz, unsupervised_score_davies_bouldin = cluster_data(data_to_cluster,ca,index_start)
                except:
                    logger.warning(traceback.format_exc())
                    logger.error("Error in clustering")
                    continue
                eval_metrics_dict = evaluate_clusters(clusterwise_indices_str,gt_lines,n_clus,'',out_dir,n_true_clusters,gt_names,main_results_dir,'',0)
                logger.debug("Evaluated clustering succesfully")
                eval_metrics_dict['Silhouette score'] = unsupervised_score_silhouette
                eval_metrics_dict['Calinski-Harabasz score'] = unsupervised_score_calinski_harabasz
                eval_metrics_dict['Davies-Bouldin score'] = unsupervised_score_davies_bouldin
                eval_metrics_dict['clustering method'] = ca
                # Save to file and use for warm start
                with open(main_results_dir + '/' + out_dir + '/eval_metrics_dict.pkl','wb') as f:
                    pkl.dump(eval_metrics_dict,f)
            if len(results_df) == 0:
                results_df = pd.DataFrame(columns = eval_metrics_dict.keys())
            results_df = results_df.append(pd.Series(eval_metrics_dict,name = embedding_method + ' ' + str(graph_embedding_method) + ' '+ node_attribute_method + ' embedding ' +  cluster_method_str + ' clustering'))

    if ('FMM F1 score' in eval_metrics_dict):
        results_df.sort_values(by='FMM F1 score',ascending=False,inplace=True)
        
    best_method = results_df.index[0]
    logger.info('Best method {}',best_method)
    best_method_sklearn = results_df['clustering method'][best_method]
    
    # Read other embedding files and append
    results_other_embeddings_path = main_results_dir + '/' + out_dir_orig + '/hyperparameter_opt_all_methods_sorted_' + dataset + '.csv'
    if os.path.exists(results_other_embeddings_path):
        logger.info('Reading and concatenating existing Hyperparameter results...')
        results_other_embeddings = pd.read_csv(results_other_embeddings_path,index_col=0)
        results_df = pd.concat([results_other_embeddings,results_df])    
        
    if ('FMM F1 score' in eval_metrics_dict):
        results_df.sort_values(by='FMM F1 score',ascending=False,inplace=True)        
    
    results_df.to_csv(main_results_dir + '/' + out_dir_orig + '/hyperparameter_opt_all_methods_sorted_' + dataset + '.csv')
    
    return best_method_sklearn,best_method
          

def main():
    # Main driver
    
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--graph_names", nargs='+', default=[""], help="Name of slicem graph, specify as list, ex: '','slicem_edge_list_l1','slicem_edge_list_euclidean','slicem_edge_list_cosine'")
    parser.add_argument("--graph_types", nargs='+', default=[""],help="Type of graph - directed, undirected or both")    
    parser.add_argument("--datasets", nargs='+', default=["real"], help="Dataset name, opts: real, synthetic, synthetic_noisy, synthetic_more_projs")
    parser.add_argument("--out_dir_suffixes", nargs='+', default=[''], help="Suffix of output directory: ex: '', '_siamese_node_embedding', '_combined_externally','_combined_internally' ")
    parser.add_argument("--node_attribute_methods", nargs='+', default=[''], help="Image embeddings used as node attributes in the graph embeddings, ex: '', 'resnet-18', 'densenet','vgg','alexnet','siamese_more_projs_all','siamese','efficientnet_b1','efficientnet_b7'")
    parser.add_argument("--graph_embedding_methods", nargs='+', default=[''], help="Image embeddings used as node attributes in the graph embeddings, ex: '','metapath2vec','wys','graphWave','node2vec' ")
    parser.add_argument("--embedding_methods", nargs='+', default=['efficientnet_b1'], help="Image embeddings - either pure or pure slicem graph, ex: 'attri2vec','gcn','cluster_gcn','gat','APPNP','graphSage' OR 'siamese_noisy','alexnet','densenet','resnet-18', 'vgg','siamese','efficientnet_b1','efficientnet_b7','siamese_more_projs_all'")
    parser.add_argument("--eval_SLICEM", default=0, help="Evaluate SLICEM results")
    parser.add_argument("--main_results_dir", default="../results", help="Main directory containing results, ex: '.' ")
    parser.add_argument("--find_best_clustering_method", default=1, help="Hyperparameter tuning to find best clustering method and parameters")
    parser.add_argument("--clustering_methods", default=[DBSCAN(),OPTICS(),Birch(n_clusters=None), AffinityPropagation()], help="Clustering methods and parameters to use, ex: DBSCAN(),MeanShift(),OPTICS(),Birch(n_clusters=None), AffinityPropagation()")
    
    
    args = parser.parse_args()
    
    # use clustering method if specified to do so
    if not args.find_best_clustering_method:
        best_clustering_methods = [(meth, str(meth)) for meth in args.clustering_methods]


    graph_names = args.graph_names
    
    graph_types = args.graph_types
    
    datasets = args.datasets    
    print(datasets)

    out_dir_suffixes = args.out_dir_suffixes
    
    graph_embedding_methods = args.graph_embedding_methods

    node_attribute_methods = args.node_attribute_methods
    
    #embedding_methods = ['slicem-graph-' + graph_embedding_method for graph_embedding_method in graph_embedding_methods]
    
    embedding_methods = args.embedding_methods    
 
    n_emb = len(embedding_methods)
    n_graph_emb = len(graph_embedding_methods)
    
    graph_embedding_methods = graph_embedding_methods*n_emb
    embedding_methods = list(np.repeat(embedding_methods,n_graph_emb))
        
    
    eval_SLICEM = int(args.eval_SLICEM)
    print(eval_SLICEM)

    # Hyper-parameter ranges for cross-validation
    # eps, default=0.5, The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    #cluster_hyper_param_ranges = {"DBSCAN": {"eps":"infer","min_samples":range(2,10)}}
    #cluster_hyper_param_ranges = {"DBSCAN": {"eps":np.arange(0.1,1,0.1),"min_samples":range(2,10)}}
    cluster_hyper_param_ranges = {"DBSCAN": 
                                      {"eps":np.arange(0.25,3,0.25),
                                        "min_samples":range(2,10)},
                                  "OPTICS":
                                      {"max_eps":np.arange(0.25,3,0.25),
                                        "min_samples":range(2,10)},
                                  "Birch":
                                      {"threshold":np.arange(0.1,1,0.1),
                                        "branching_factor": range(10,100,10),
                                        "n_clusters": [None]},
                                  "AffinityPropagation":
                                      {"damping":np.arange(0.5,1,0.1),"random_state":[7]}
                                  }
    
    # To do: Add mean shift

    main_results_dir = args.main_results_dir
    
    
    for graph_name in graph_names:
        for graph_type in graph_types:
            # experiment_name = ''
            if graph_name == 'slicem_edge_list_l1':
                graph_name_exp = 'slicem_l1'
            elif graph_name == 'slicem_edge_list_euclidean':
                graph_name_exp = 'slicem_l2'
            elif graph_name == 'slicem_edge_list_cosine':
                graph_name_exp = 'slicem_cosine'   
            else:
                graph_name_exp = graph_name
                
            experiment_name = graph_name_exp + '_' + graph_type    
            
            config = dict()
        
             
            if not os.path.exists(main_results_dir):
                os.mkdir(main_results_dir)
                
            for out_dir_suffix in out_dir_suffixes:
                if out_dir_suffix == '_combined_internally':        
                    combine_graph_flag = 0
                    combine_graph_flag_internal= 1
                elif out_dir_suffix == '_combined_externally':
                    combine_graph_flag = 1            
                    combine_graph_flag_internal=0    
                else:
                    combine_graph_flag = 0            
                    combine_graph_flag_internal=0               
                    
                for var in [combine_graph_flag,combine_graph_flag_internal,graph_embedding_methods,embedding_methods,datasets,cluster_hyper_param_ranges,main_results_dir]:
                    my_var_name = [ k for k,v in locals().items() if v is var][0]
                    config[my_var_name] = var            
            
                for dataset in datasets:
                    images_file_name,images_true_labels,sep,index_start,out_dir_orig,sep2 = get_config(dataset)
                    out_dir_orig = out_dir_orig + out_dir_suffix + '_'+experiment_name
            
                    for var in [images_file_name,images_true_labels,sep,index_start,out_dir_orig]:
                        my_var_name = [ k for k,v in locals().items() if v is var][0]
                        config[my_var_name] = var      
                    
                    results_dir = main_results_dir + '/' + out_dir_orig
                    if not os.path.exists(results_dir):
                        os.mkdir(results_dir)
                        
                    with open(results_dir + "/config.yaml", 'a+') as outfile:
                        yaml_dump(config, outfile, default_flow_style=False)               
                        
                    # Setting logger
                    logger.add(results_dir + '/log_file.txt',level="INFO")
                        
                    data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep, sep2)
                    n_true_clusters = len(gt_lines)
                            
                    results_df = pd.DataFrame()
                    test_results_df = pd.DataFrame()
                    
                    embedding_eval_df = pd.DataFrame()
                    
                    # Split into train and test sets
                    
                    train_cluster_inds, test_cluster_inds, train_clusters, test_clusters = train_test_split_complexes(gt_lines)
                    with open(results_dir + '/train_clusters.txt','w') as f:
                        f.writelines([' '.join(list(comp)) + '\n' for comp in train_clusters])
                    with open(results_dir+ '/test_clusters.txt','w') as f:
                        f.writelines([' '.join(list(comp)) + '\n' for comp in test_clusters])            
                    
                    if eval_SLICEM:            
                        eval_metrics_dict_SLICEM = evaluate_SLICEM(gt_lines,gt_names,n_true_clusters,dataset,sep,index_start)

                    for i,embedding_method in enumerate(embedding_methods):
                        graph_embedding_method = graph_embedding_methods[i]
                        logger.info(embedding_method)
                                                
                        for node_attribute_method in node_attribute_methods:
                            out_dir_emb = out_dir_orig + '/'+embedding_method + str(graph_embedding_method)+node_attribute_method
                            if not os.path.exists(main_results_dir + '/' + out_dir_emb):
                                os.mkdir(main_results_dir + '/' + out_dir_emb)
                            data_to_cluster = get_image_embedding(data,embedding_method,combine_graph_flag_internal,graph_embedding_method,dataset,graph_name,graph_type,node_attribute_method)
                            
                            data_to_cluster = reduce_dimensions(data_to_cluster)
                            
                            # Save reduced dimension embeddings of the images 
                            with open(main_results_dir + '/' + out_dir_emb + '/' + embedding_method + '_reduced_embeddings.npy', 'wb') as f:
                                np.save(f, data_to_cluster)
                            
                            # Get graph embeddings and combine 
                            if combine_graph_flag:
                                if graph_embedding_method in ['attri2vec','gcn','cluster_gcn','gat','APPNP','graphSage']:
                                    graph_vectors = slicem_graph_embeddings(dataset,graph_embedding_method,'stellar',graph_name,'siamese',graph_type)
                                else:
                                    graph_vectors = slicem_graph_embeddings(dataset,graph_embedding_method,'stellar',graph_name,'',graph_type)
                                    
                                graph_vectors = reduce_dimensions(graph_vectors)
                                
                                data_to_cluster = np.hstack((data_to_cluster,graph_vectors))  
                            
                            image_wise_cluster_labels = get_image_wise_cluster_labels(data_to_cluster,gt_lines,index_start)
                            
                            train_image_wise_cluster_labels, train_vectors, test_image_wise_cluster_labels, test_vectors, train_images_orig_names, test_images_orig_names = get_train_images(image_wise_cluster_labels,train_cluster_inds,data_to_cluster,index_start)
                            
                            train_cluster_names = [gt_names[ind] for ind in train_cluster_inds]
                            test_cluster_names = [gt_names[ind] for ind in test_cluster_inds]
                            
                            train_cluster_array = np.vstack(train_vectors)
                            test_cluster_array = np.vstack(test_vectors)
                                        
                            silhouette_dict_full = evaluate_embeddings(data_to_cluster, image_wise_cluster_labels)
                            
                            silhouette_dict = evaluate_embeddings(train_vectors, train_image_wise_cluster_labels)
                            
                            if len(embedding_eval_df) == 0:
                                embedding_eval_df = pd.DataFrame(columns = silhouette_dict.keys())    
                                
                            embedding_eval_df = embedding_eval_df.append(pd.Series(silhouette_dict,name = embedding_method + ' ' + str(graph_embedding_method)+ ' with train data'))
                            embedding_eval_df = embedding_eval_df.append(pd.Series(silhouette_dict_full,name = embedding_method+ ' ' + str(graph_embedding_method) + ' with full data'))
                
                            # Plot TSNEs
                            try:
                                plot_tsne(data_to_cluster,out_dir_emb,image_wise_cluster_labels, gt_names, silhouette_dict['max_silhouette_distance'] ,main_results_dir)                
                            except Exception as e: # Add traceback
                                logger.error('ERROR in tsne plot:')
                                logger.error(str(e))
                            
                            # Set distance threshold based on PCA variance
                                                        
                            if args.find_best_clustering_method:
                                # Training clustering hyper parameters using training data
                                
                                best_method,best_method_name = cluster_hyperparameter_optimization(cluster_hyper_param_ranges,train_cluster_array,train_image_wise_cluster_labels,index_start,embedding_method,train_clusters,train_cluster_names,n_true_clusters,out_dir_orig,dataset,silhouette_dict['max_silhouette_distance'],main_results_dir,graph_embedding_method,node_attribute_method)
                    
                                # Final clustering on full data using the best method and parameters
                                best_clustering_methods = [(best_method,best_method_name)]
                            
                            for clustering_method_tup in best_clustering_methods:
                                clustering_method,clustering_method_name = clustering_method_tup
                                clustering_method_name = clustering_method_name.split('embedding')[1].split('clustering')[0].rstrip()
                                out_dir = out_dir_emb + '/'+str(clustering_method_name)  
                                if not os.path.exists(main_results_dir + '/' + out_dir):
                                    os.mkdir(main_results_dir + '/' + out_dir)
                                    
                                #cluster_hyperparameter_optimization_clusteval(data_to_cluster,clustering_method)
                                                 
                                n_clus, clusterwise_indices_str,unsupervised_score_silhouette,unsupervised_score_calinski_harabasz, unsupervised_score_davies_bouldin = cluster_data(data_to_cluster,clustering_method,index_start,silhouette_dict['max_silhouette_distance'])
                                write_clusters(clusterwise_indices_str,'',out_dir,main_results_dir)                
                                
                                # On full data
                                eval_metrics_dict = evaluate_clusters(clusterwise_indices_str,gt_lines,n_clus,'',out_dir,n_true_clusters,gt_names,main_results_dir)
                                eval_metrics_dict['Silhouette score'] = unsupervised_score_silhouette
                                eval_metrics_dict['Calinski-Harabasz score'] = unsupervised_score_calinski_harabasz
                                eval_metrics_dict['Davies-Bouldin score'] = unsupervised_score_davies_bouldin
                                
                                if len(results_df) == 0:
                                    results_df = pd.DataFrame(columns = eval_metrics_dict.keys())
                                eval_metrics_dict['Image Embedding Method']  = embedding_method 
                                eval_metrics_dict['Graph Node Embedding Method']  = str(graph_embedding_method) + node_attribute_method
                                eval_metrics_dict['Clustering Method']  = str(clustering_method).split('(')[0]
                                eval_metrics_dict['Clustering Parameters']  = str(clustering_method).split('(')[1][:-1]
                                
                                results_df = results_df.append(pd.Series(eval_metrics_dict,name = embedding_method + ' ' + str(graph_embedding_method) +' '+ node_attribute_method +  ' embedding ' +  str(clustering_method) + ' clustering'))
                                
                                # On test data
                                n_clus, clusterwise_indices_str,unsupervised_score_silhouette,unsupervised_score_calinski_harabasz, unsupervised_score_davies_bouldin = cluster_data(test_cluster_array,clustering_method,index_start,silhouette_dict['max_silhouette_distance'],test_images_orig_names)
                                
                                test_eval_metrics_dict = evaluate_clusters(clusterwise_indices_str,test_clusters,n_clus,'test',out_dir,len(test_clusters),test_cluster_names,main_results_dir)
                                test_eval_metrics_dict['Silhouette score'] = unsupervised_score_silhouette
                                test_eval_metrics_dict['Calinski-Harabasz score'] = unsupervised_score_calinski_harabasz
                                test_eval_metrics_dict['Davies-Bouldin score'] = unsupervised_score_davies_bouldin
                                
                                if len(results_df) == 0:
                                    test_results_df = pd.DataFrame(columns = test_eval_metrics_dict.keys())
                                test_eval_metrics_dict['Image Embedding Method']  = embedding_method 
                                test_eval_metrics_dict['Graph Node Embedding Method']  = str(graph_embedding_method) + node_attribute_method
                                test_eval_metrics_dict['Clustering Method']  = str(clustering_method).split('(')[0]
                                test_eval_metrics_dict['Clustering Parameters']  = str(clustering_method).split('(')[1][:-1]

                                test_results_df = test_results_df.append(pd.Series(test_eval_metrics_dict,name = embedding_method + ' ' + str(graph_embedding_method) +' '+ node_attribute_method +  ' embedding ' +  str(clustering_method) + ' clustering'))
                
                    if eval_SLICEM:            
                        results_df = results_df.append(pd.Series(eval_metrics_dict_SLICEM,name = 'SLICEM'))
                        
                    #results_df.sort_values(by='No. of clusters',key=lambda x: abs(x-n_true_clusters),inplace=True)
                    if ('FMM F1 score' in eval_metrics_dict):
                        results_df.sort_values(by='FMM F1 score',ascending=False,inplace=True)
                        
                    if ('FMM F1 score' in test_eval_metrics_dict):
                        test_results_df.sort_values(by='FMM F1 score',ascending=False,inplace=True) 
                        
                    if ('FMM F1 score w/o junk' in eval_metrics_dict):
                        results_df.sort_values(by='FMM F1 score w/o junk',ascending=False,inplace=True)
                        
                    if ('FMM F1 score w/o junk' in test_eval_metrics_dict):
                        test_results_df.sort_values(by='FMM F1 score w/o junk',ascending=False,inplace=True)                             

                    embedding_eval_df = embedding_eval_df.round(3)
                    test_results_df = test_results_df.round(3)
                    results_df = results_df.round(3)
                    embedding_eval_df.to_csv(main_results_dir + '/' + out_dir_orig + '/evaluating_embeddings_' + dataset + '.csv')
                    
                    test_results_df.to_csv(main_results_dir + '/' + out_dir_orig + '/compiled_test_results_all_methods_sorted_' + dataset + '.csv')
                    
                    results_df.to_csv(main_results_dir + '/' + out_dir_orig + '/compiled_results_all_methods_sorted_' + dataset + '.csv')
    
        
if __name__ == "__main__":    
    main()
