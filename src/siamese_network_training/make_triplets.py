# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 04:04:27 2021

@author: Meghana
"""

from cluster_image_embeddings import get_config, read_data

images_file_name,images_true_labels,sep,index_start,out_dir = get_config('synthetic')

data, gt_lines, gt_names = read_data(images_file_name, images_true_labels, sep)