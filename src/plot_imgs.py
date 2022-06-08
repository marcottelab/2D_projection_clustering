# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:05:20 2022

@author: Meghana
"""

from util.input_functions import get_config, read_data
from cluster_image_embeddings import get_image_wise_cluster_labels
from matplotlib import pyplot as plt

import os

datasets = ['synthetic_noisy']

#dataset = 'synthetic'
#dataset = 'synthetic_more_projs'

for dataset in datasets:
    
    dir_name = '../data/'+dataset+'_dataset' + '/particle_imgs'
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    images_file_name,images_true_labels,sep,index_start,out_dir_orig,sep2 = get_config(dataset)
    
    data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep, sep2)
    
    image_wise_cluster_labels = get_image_wise_cluster_labels(data,gt_lines,index_start)
    image_wise_cluster_label_names = [gt_names[clus] for clus in image_wise_cluster_labels]
    
    data_nonneg = []
    for myarray in data:
        myarray[myarray <= 0] = 0
        data_nonneg.append(myarray)
        
    data = data_nonneg
    
    plt.gray()
    for num in range(len(data)):
        fig = plt.figure(figsize=(4,4))
        ax = plt.axes(frameon=False, xticks=[],yticks=[])
        ax.imshow(data[num])
        
        plt.title(image_wise_cluster_label_names[num])
        plt.savefig(dir_name+'/img'+str(num) +'.png', bbox_inches='tight', pad_inches=0)
    
    #plt.imsave(dir_name+'/img'+str(num) +'.png',data[num])