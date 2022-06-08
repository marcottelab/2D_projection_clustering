# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:05:20 2022

@author: Meghana
"""

from util.input_functions import get_config, read_data
from matplotlib import pyplot as plt

import os

#dataset = 'synthetic'
dataset = 'synthetic_more_projs'

dir_name = '../data/'+dataset+'_dataset' + '/particle_imgs'

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    
images_file_name,images_true_labels,sep,index_start,out_dir_orig,sep2 = get_config(dataset)

data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep, sep2)

data_nonneg = []
for myarray in data:
    myarray[myarray <= 0] = 0
    data_nonneg.append(myarray)
    
data = data_nonneg

plt.gray()
num = 0
plt.imsave(dir_name+'/img'+str(num) +'.png',data[num])