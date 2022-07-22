# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 18:26:07 2022

@author: Abdur Rehman
"""

# %%
from matplotlib import gridspec
import time
import matplotlib.pyplot as plt
import pandas as pd
from keras.metrics import MeanSquaredError
from casadi import *
import sys
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import Input, layers
from keras_tuner import RandomSearch
import keras_tuner as kt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
from tensorflow.keras import regularizers

print('Importing the libraries')

# %%
print('Importing PGNN libraries')
# Importing the libraries
from tf_pgnn import PGNN

# %%
# %%
print('Importing Custom Inputs and Outputs Function Being Called')
from Machine_Learning_NARX import X_train_HyperParametersModel, X_test_HyperParametersModel, y_train_HyperParametersModel, y_test_HyperParametersModel
# %%
def network_derivative_loss(params_1, params_2):
    # X_HyperParametersModel = params_1
    # y_closed_loop_PGNN = closed_loop_function(params_2) 
    # y_closed_loop_PGNN = tf.constant(y_closed_loop_PGNN, dtype=tf.float32)
    # must be defined with these input arguments - template must be followed for this to work
    tic_closed_loop = time.perf_counter()  # Start Time
    print('Starting Time BABA:', tic_closed_loop)
    
    # Needs to make sure the model and the inputs are built on the same parameters
    inputs = params_2
    
    # Finding out the rows and columns for the closed loop simulation array
    row_closed_loop, column_closed_loop = inputs.shape
    
    # Creation of the Closed Loop array
    #closed_loop_NParray = np.zeros((row_closed_loop+1, column_closed_loop))
    closed_loop_NParray = np.zeros((row_closed_loop, column_closed_loop))
    
    # Setting up the initail Valuse for the closed loop simulation from the inputs[0] to dummy variable [0]
    dummy_0 = inputs.iloc[0]
    closed_loop_NParray[0] = dummy_0
    
    # Defining the rows and columns of the closed loop simulation array
    row_iteration = 0
    column_iteration = 0
    
    # Fining the delays nU=nY from the NARX Input
    iteration = int(column_closed_loop/6)
    iteration_var = 0
    
    # Finding the start point of the inputs in an array
    input_start = column_closed_loop - iteration * 2
    #
    
    
    def custom_physics_loss(X_train_HyperParametersModel, y_train_HyperParametersModel, model):
        """
        Here we will use the physics information (differential equation) to steer the network towards making
        correct predictions for unseen / unlabelled data.
        """
        # bring the closed loop model here
        #y_test_predict_PGNN = model.predict(X_HyperParametersModel)
        #y_closed_loop_PGNN = closed_loop_function(X_HyperParametersModel_Pandas_Dataframe) 
        # compute the first derivative of predicted height wrt current input using automatic differentiation
        # with tf.GradientTape() as tape:
        #     tape.watch(X_test_HyperParametersModel)
        #     h_pred = model.predict(X_test_HyperParametersModel)  # compute the neural network output for collocation points X_test_HyperParametersModel

        # dh = tape.gradient(h_pred, X_test_HyperParametersModel)
        # dh_du = dh[:, 3]  # the gradient with respect to the input is in the third column

        # dont get negative constraints for Ca and CB
        # evaluate this on closed loop predictions

        #loss = tf.math.reduce_sum(tf.nn.relu(-(y_closed_loop_PGNN[:, 3] - y_test_predict_PGNN[:, 3])))
        #loss = tf.math.reduce_sum((y_closed_loop_PGNN[:, 3] - y_test_predict_PGNN[:, 3]))
        #udendiff = -y_closed_loop_PGNN[:, 3] # ReLU(-C_A)
        print('START')
        for row_iteration in range(row_closed_loop - 1):
            print('START')
            # Column Iteration
            column_iteration = 0
            print('2')
            #for column_iteration in range(1):
            while column_iteration < 1:
                print('3')
                # Taking the entire row and feeding it to the Neural Network (X_K)
                states_from_the_model = closed_loop_NParray[row_iteration]
                states_from_the_model = numpy.reshape(
                    states_from_the_model, (1, column_closed_loop))
                states_from_the_model = model.predict(states_from_the_model)
                print(states_from_the_model)
                
                # Shifting all of the STATES based on the delay (X_K-1)
                closed_loop_NParray[row_iteration+1,
                                    4:input_start] = closed_loop_NParray[row_iteration, 0: input_start - 4]
        
                # Transferring the Neural Network States to the Closed Loop Simulating Array
                closed_loop_NParray[row_iteration+1, 0:4] = states_from_the_model
        
                # Shifting all of the Inputs based on the delay (U_k-1)
        
                closed_loop_NParray[row_iteration+1, input_start +
                                    2:column_closed_loop] = closed_loop_NParray[row_iteration, input_start:column_closed_loop-2]
        
                # Importing the Inputs to the Closed Loop Simulation (U_k)
                closed_loop_NParray[row_iteration+1, input_start:input_start +
                                    2] = inputs.iloc[row_iteration+1, column_closed_loop-2:column_closed_loop-1]
        
                column_iteration += 1
        
            row_iteration += 1
            print('END')
        
        
        #y_test_predict_PGNN = model.predict(X_train_HyperParametersModel)
        print(y_test_predict_PGNN)
        udendiff = -closed_loop_NParray[:, 1]
        #loss = tf.math.reduce_sum(tf.nn.relu(1))
        loss = tf.math.reduce_sum(tf.nn.relu(udendiff))
        return loss

    return custom_physics_loss


# %%
#start = time.process_time()


def closed_loop_function(parameters, model):
#
    tic_closed_loop = time.perf_counter()  # Start Time
    print('Starting Time:', tic_closed_loop)
    # Closed Loop Simulation
    print('Closed Loop Simulation Started')
    
    # Needs to make sure the model and the inputs are built on the same parameters
    inputs = parameters
    history_model = model
    
    # Finding out the rows and columns for the closed loop simulation array
    row_closed_loop, column_closed_loop = inputs.shape
    
    # Creation of the Closed Loop array
    #closed_loop_NParray = np.zeros((row_closed_loop+1, column_closed_loop))
    closed_loop_NParray = np.zeros((row_closed_loop, column_closed_loop))
    
    # Setting up the initail Valuse for the closed loop simulation from the inputs[0] to dummy variable [0]
    dummy_0 = inputs.iloc[0]
    closed_loop_NParray[0] = dummy_0
    
    # Defining the rows and columns of the closed loop simulation array
    row_iteration = 0
    column_iteration = 0
    
    # Fining the delays nU=nY from the NARX Input
    iteration = int(column_closed_loop/6)
    iteration_var = 0
    
    # Finding the start point of the inputs in an array
    input_start = column_closed_loop - iteration * 2
    
    # Row Iteration
    for row_iteration in range(row_closed_loop - 1):
    
        # Column Iteration
        column_iteration = 0
    
        #for column_iteration in range(1):
        while column_iteration < 1:
    
            # Taking the entire row and feeding it to the Neural Network (X_K)
            states_from_the_model = closed_loop_NParray[row_iteration]
            states_from_the_model = numpy.reshape(
                states_from_the_model, (1, column_closed_loop))
            states_from_the_model = history_model.predict(states_from_the_model)
    
            # Shifting all of the STATES based on the delay (X_K-1)
            closed_loop_NParray[row_iteration+1,
                                4:input_start] = closed_loop_NParray[row_iteration, 0: input_start - 4]
    
            # Transferring the Neural Network States to the Closed Loop Simulating Array
            closed_loop_NParray[row_iteration+1, 0:4] = states_from_the_model
    
            # Shifting all of the Inputs based on the delay (U_k-1)
    
            closed_loop_NParray[row_iteration+1, input_start +
                                2:column_closed_loop] = closed_loop_NParray[row_iteration, input_start:column_closed_loop-2]
    
            # Importing the Inputs to the Closed Loop Simulation (U_k)
            closed_loop_NParray[row_iteration+1, input_start:input_start +
                                2] = inputs.iloc[row_iteration+1, column_closed_loop-2:column_closed_loop-1]
    
            column_iteration += 1
    
        row_iteration += 1
    
    
    toc_closed_loop = time.perf_counter()  # End Time
    print(
        f"Build finished in  closed_loop {(toc_closed_loop - tic_closed_loop)/60:0.0f} minutes {(toc_closed_loop - tic_closed_loop)%60:0.0f} seconds")
    print('Closed Loop Simulation Ended')

    return closed_loop_NParray


        # bring the closed loop model here
        #y_test_predict_PGNN = model.predict(X_HyperParametersModel)
        #y_closed_loop_PGNN = closed_loop_function(X_HyperParametersModel_Pandas_Dataframe) 
        # compute the first derivative of predicted height wrt current input using automatic differentiation
        # with tf.GradientTape() as tape:
        #     tape.watch(X_test_HyperParametersModel)
        #     h_pred = model.predict(X_test_HyperParametersModel)  # compute the neural network output for collocation points X_test_HyperParametersModel

        # dh = tape.gradient(h_pred, X_test_HyperParametersModel)
        # dh_du = dh[:, 3]  # the gradient with respect to the input is in the third column

        # dont get negative constraints for Ca and CB
        # evaluate this on closed loop predictions

        #loss = tf.math.reduce_sum(tf.nn.relu(-(y_closed_loop_PGNN[:, 3] - y_test_predict_PGNN[:, 3])))
        #loss = tf.math.reduce_sum((y_closed_loop_PGNN[:, 3] - y_test_predict_PGNN[:, 3]))
        #udendiff = -y_closed_loop_PGNN[:, 3] # ReLU(-C_A)