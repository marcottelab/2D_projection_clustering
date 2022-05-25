# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:13:11 2021

@author: Meghana
"""

from tensorflow.keras.applications import resnet
from loguru import logger
import numpy as np
import tensorflow as tf


def siamese_embedding(list_of_PIL_imgs,embedding_model = 'siamese'):
    '''
    Get image embeddings using a trained siamese neural network

    Parameters
    ----------
    list_of_PIL_imgs : list[]
        list of PIL image objects corresponding to each image in the data

    Returns
    -------
    embedding_npy : numpy.ndarray
        Array of image embeddings from a trained siamese network

    '''
    converted_data = tf.convert_to_tensor([tf.keras.preprocessing.image.img_to_array(image_PIL) for image_PIL in list_of_PIL_imgs])
    preprocessed_data = resnet.preprocess_input(converted_data)
    # #model_name = '../models/siamese_embedding_model_preprocessed_more_negs.tf'
    if embedding_model == 'siamese':
        model_name = '../models/siamese_embedding_model.tf'
    elif embedding_model == 'siamese_noisy':
        model_name = '../models/synthetic_noisysiamese_embedding_model.tf'        
    elif embedding_model == 'siamese_more_projs_noisy':
        model_name = '../models/synthetic_more_projs_noisysiamese_embedding_model.tf'        
    elif embedding_model == 'siamese_real':
        model_name = '../models/realsiamese_embedding_model.tf'        
    else:
        model_name = '../models/synthetic_more_projssiamese_embedding_model.tf'        
    embedding = tf.keras.models.load_model(model_name)
    logger.info('Loaded model')    
    
    data_embedding = embedding(preprocessed_data)
    
    embedding_npy = data_embedding.numpy()
    return embedding_npy
