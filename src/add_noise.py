# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:05:20 2022

@author: Meghana
"""
from util.input_functions import get_config, read_data
from matplotlib import pyplot as plt
from argparse import ArgumentParser as argparse_ArgumentParser

import skimage
import numpy as np
import mrcfile


def main():
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--dataset_type", default="synthetic_more_projs", help="Dataset name, opts: synthetic")
    
    args = parser.parse_args()
    
    dataset = args.dataset_type
    images_file_name,images_true_labels,sep,index_start,out_dir_orig,sep2 = get_config(dataset)
    
    data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep, sep2)
    
    data_nonneg = []
    for myarray in data:
        myarray[myarray <= 0] = 0
        data_nonneg.append(myarray)
        
    data = data_nonneg
    
    plt.gray()
    plt.imsave('../data/'+dataset+'_noisy_dataset/img1_original.png',data[0])
    
    plt.imsave('../data/'+dataset+'_noisy_dataset/img2_original.png',data[1])
    
    dims = data[0].shape
    low_dim, high_dim = dims
    
    # Add gaussian noise
    data_noisy = []
    for myarray in data:
        mn = np.mean(myarray)
        vr = np.var(myarray)
        myarray = skimage.util.random_noise(myarray, mode='gaussian', seed=None, clip=False, mean = mn, var = vr)
        
        # Add hot and dead pixels
        myarray = skimage.util.random_noise(myarray, mode='s&p', seed=None, amount = 0.01, clip = False)
        
        data_noisy.append(myarray)
    
    data_noisy_nonneg = []
    for myarray in data_noisy:
        myarray[myarray <= 0] = 0
        data_noisy_nonneg.append(myarray)
        
    plt.imsave('../data/'+dataset+'_noisy_dataset/img1_noisy.png',data_noisy_nonneg[0])
    
    plt.imsave('../data/'+dataset+'_noisy_dataset/img2_noisy.png',data_noisy_nonneg[1])
    
    data_noisy_arr = np.array(data_noisy_nonneg, dtype='<f4')
    
    with mrcfile.new('../data/'+dataset+'_noisy_dataset/'+dataset+'_noisy.mrcs',overwrite=True) as mrc:
          mrc.set_data(data_noisy_arr)


if __name__ == "__main__":
    main()