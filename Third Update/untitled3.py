# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:49:08 2022

@author: Abdur Rehman
"""

# imports
import logging
import matplotlib

matplotlib.use('Qt5Agg')
import tensorflow as tf

tf.config.run_functions_eagerly(True)

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# %%
from tf_pgnn import PGNN
# %%
from helper_functions import  get_working_directory, create_results_folder, create_logger


# %% load the lake data
working_directory = helper_functions.get_working_directory()

use_YPhy = 0  # Whether YPhy is used as another feature in the NN model or not

lake = ['mendota', 'mille_lacs']
lake_num = 0  # 0 : mendota , 1 : mille_lacs
lake_name = lake[lake_num]

# Loading supervised learning data
# Load features (Xc) and target values (Y)
data_dir = working_directory + '/datasets/'
filename = lake_name + '.mat'
mat = spio.loadmat(data_dir + filename, squeeze_me=True,
                   variable_names=['Y', 'Xc_doy', 'Modeled_temp'])
Xc = mat['Xc_doy']
Y = mat['Y']

# Loading unsupervised data
unsup_filename = lake_name + '_sampled.mat'
unsup_mat = spio.loadmat(data_dir + unsup_filename, squeeze_me=True,
                         variable_names=['Xc_doy1', 'Xc_doy2'])

uX1 = unsup_mat['Xc_doy1']  # Xc at depth i for every pair of consecutive depth values
uX2 = unsup_mat['Xc_doy2']  # Xc at depth i + 1 for every pair of consecutive depth values

if use_YPhy == 0:
    # Removing the last column from uX (corresponding to Y_PHY)
    # uX1 = uX1[:, :-1]
    # uX2 = uX2[:, :-1]
    # regularization terms
    lambda_physics = 0  # Physics-based regularization constant
else:
    lambda_physics = 1000 * 0.5  