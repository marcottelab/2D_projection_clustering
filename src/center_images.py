# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 23:04:51 2022

@author: Meghana
"""

from slicem import get_projection_2D
import mrcfile
import numpy as np

out_path =  "../data/synthetic_more_projections/"
#mrcs_file = "../data/synthetic_raw_projections/1a0i/2D_projection1.mrc"
#mrcs_file = "../data/synthetic_more_projections/5DPAPAPAPTPGPCPCPTPGPGPTPCPT3_1I6H_proj.mrcs"
mrcs_file = out_path+"16SRIBOSOMALRNA_1HNW_proj.mrcs"


shape, projection_2D = get_projection_2D(mrcs=mrcs_file, factor=1,out_size=(350,350),resize=True)
#shape, projection_2D = get_projection_2D(mrcs=mrcs_file, factor=2)
projection_2D_arr = np.array(list(projection_2D.values()))

with mrcfile.new(mrcs_file+'_processed.mrcs',overwrite=True) as mrc:
     mrc.set_data(projection_2D_arr)