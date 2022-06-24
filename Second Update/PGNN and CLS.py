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
print('Importing Custom Inputs (X_train and X_test ) and Outputs (y_train and y_test) Function Being Called')
X_train_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/X_train_HyperParametersModel.xlsx")
X_test_HyperParametersModel = pd.read_excel("C:/Users/Abdur Rehman/Thesis Code/X_test_HyperParametersModel.xlsx")
y_train_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/y_train_HyperParametersModel.xlsx")
y_test_HyperParametersModel = pd.read_excel("C:/Users/Abdur Rehman/Thesis Code/y_test_HyperParametersModel.xlsx")

# %%
rows_CLS, columns_CLS = X_test_HyperParametersModel.shape
# %%
print(' Importing the Best Hyperparameters Values from the Vanialla Netwrok ')

best_hp_Dataframe = pd.read_excel("C:/Users/Abdur Rehman/Thesis Code/best_hp_Dataframe.xlsx")

# %%
print(' Extracting the values from the Best Hyperparameters variable')

input_delay_size = best_hp_Dataframe['input_delay_size']
#input_delay_size = int(4)

activation_ = best_hp_Dataframe['activation_']
#activation_ = str(relu)

learning_rate = best_hp_Dataframe['learning_rate']
#learning_rate = Float(0.1)

units_nodes_ = best_hp_Dataframe['units_nodes_']
#units_nodes_ = int(30)

hidden_layers = best_hp_Dataframe['hidden_layers']
#hidden_layers = int(3)

momentum_best = best_hp_Dataframe['momentum']
#momentum_best = Float(0.5)

batch_size_best = best_hp_Dataframe['batch_size']
#batch_size_best = int(64)


# %%
print('PGNN with physics_loss_function')

tic_closed_loop = time.perf_counter()  # Start Time
toc_closed_loop = time.perf_counter()  # Start Time

print(
    f"Build finished in {(toc_closed_loop - tic_closed_loop)/60:0.0f} minutes {(toc_closed_loop - tic_closed_loop)%60:0.0f} seconds")

print('Starting Time', tic_closed_loop)
# %%


def network_derivative_loss(params_1, params_2):
    # must be defined with these input arguments - template must be followed for this to work
    # Needs to make sure the model and the inputs are built on the same parameters
    inputs = params_2
    input_2 = params_1
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
    #closed_loop_NParray = tf.constant(closed_loop_NParray, dtype=tf.float32)
    #print('Starting Time:', tic_closed_loop)

    def custom_physics_loss(X_train_HyperParametersModel, y_train_HyperParametersModel, model):
        """
        Here we will use the physics information (differential equation) to steer the network towards making
        correct predictions for unseen / unlabelled data.
        """
        for row_iteration in range(rows_CLS - 1):
            #print('START')
            # Column Iteration
            column_iteration = 0
            #print('2')
            #for column_iteration in range(1):
            while column_iteration < 1:
                #print('3')
                # Taking the entire row and feeding it to the Neural Network (X_K)
                
                #print('Starting Time states_from_the_model:', tic_closed_loop)

                states_from_the_model = closed_loop_NParray[row_iteration]
                states_from_the_model = numpy.reshape(
                    states_from_the_model, (1, column_closed_loop))
                states_from_the_model = model.predict(states_from_the_model)
                
                # Shifting all of the STATES based on the delay (X_K-1)
                closed_loop_NParray[row_iteration+1,
                                    4:input_start] = closed_loop_NParray[row_iteration, 0: input_start - 4]

                # Transferring the Neural Network States to the Closed Loop Simulating Array
                closed_loop_NParray[row_iteration +
                                    1, 0:4] = states_from_the_model

                # Shifting all of the Inputs based on the delay (U_k-1)
                closed_loop_NParray[row_iteration+1, input_start +
                                    2:column_closed_loop] = closed_loop_NParray[row_iteration, input_start:column_closed_loop-2]

                # Importing the Inputs to the Closed Loop Simulation (U_k)
                closed_loop_NParray[row_iteration+1, input_start:input_start +
                                    2] = inputs.iloc[row_iteration+1, column_closed_loop-2:column_closed_loop-1]

                column_iteration += 1

            row_iteration += 1

        ca_PGNN = -tf.constant(closed_loop_NParray[:][0], dtype=tf.float32)
        cb_PGNN = -tf.constant(closed_loop_NParray[:][1], dtype=tf.float32)

        loss_1 = tf.math.reduce_sum(tf.nn.relu(ca_PGNN))
        loss_2 = tf.math.reduce_sum(tf.nn.relu(cb_PGNN))
        loss  =  tf.math.add (loss_1, loss_2)
        return loss
        

    return custom_physics_loss


# %% create a pgnn
xavier = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

# using an MLP
model = PGNN(
    
    # tf.keras.Sequential(),
    
    tf.keras.layers.Flatten(input_shape=[X_train_HyperParametersModel.shape[1]]),

    # tf.keras.layers.Dense(int(input_delay_size[0]), kernel_initializer=xavier, activation=tf.nn.sigmoid,
    #                       input_shape=[X_train_HyperParametersModel.shape[1]]),

    tf.keras.layers.Dense(units_nodes_,kernel_initializer=xavier,
                          activation="sigmoid"),

    tf.keras.layers.Dense(units_nodes_, kernel_initializer=xavier,
                          activation="sigmoid"),

    tf.keras.layers.Dense(units_nodes_, kernel_initializer=xavier,
                          activation="sigmoid"),


    tf.keras.layers.Dense(4,kernel_initializer=xavier ),

    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate[0], momentum = momentum_best[0]),

    default_loss_function='mse',

    physics_loss_function=network_derivative_loss(tf.constant(
        X_test_HyperParametersModel, dtype=tf.float32), X_test_HyperParametersModel),


    lambda_default=0.5,
    lambda_physics=0.5,

    # early_stop_limit=2e-6
)

# %%
tic_closed_loop = time.perf_counter()  # Start Time
print('Starting Time:', tic_closed_loop)

model.train_model(tf.constant(X_train_HyperParametersModel, dtype=tf.float32), tf.constant(
    y_train_HyperParametersModel, dtype=tf.float32), num_epochs=10,)

toc_closed_loop = time.perf_counter()  # End Time
print('Closing Time:', tic_closed_loop)
print(
    f"Build finished in  closed_loop {(toc_closed_loop - tic_closed_loop)/60:0.0f} minutes {(toc_closed_loop - tic_closed_loop)%60:0.0f} seconds")
print('Closed Loop Simulation Ended')

# %%

get_loss_training_set = model.get_loss(tf.constant(X_train_HyperParametersModel, dtype=tf.float32), tf.constant(
    y_train_HyperParametersModel, dtype=tf.float32))
print('Loss on Training Set:', get_loss_training_set)
get_loss_testing_set = model.get_loss(tf.constant(X_test_HyperParametersModel, dtype=tf.float32), tf.constant(
    y_test_HyperParametersModel, dtype=tf.float32))
print('Loss on Testing Set:', get_loss_training_set)
# %%
# plot the model prediction for the training and testing datasets
y_pred_train = model.predict(tf.constant(
    X_train_HyperParametersModel, dtype=tf.float32)).numpy()

y_pred_test = model.predict(tf.constant(
    X_test_HyperParametersModel, dtype=tf.float32)).numpy()

# %%
time_x_axis_training = np.linspace(0, 3000, 3000)
time_x_axis_testing = np.linspace(
    3000, int(3600 - input_delay_size), int(3600 - input_delay_size - 3000),)
# time_x_axis_testing = np.linspace(
#     3000, 3583, 583)
# %%
print('the Concentration of reactant A (CA)')

#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(time_x_axis_training, y_train_HyperParametersModel[0], 'b')
plt.plot(time_x_axis_training, y_pred_train[:, 0], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data sets: the Concentration of reactant A (CA)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant A (CA)')
plt.legend(['Expected: CA', 'NN: CA'])
plt.show()
# %%
print('the Concentration of reactant B (CB)')

#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(time_x_axis_training, y_train_HyperParametersModel[1], 'b')
plt.plot(time_x_axis_training, y_pred_train[:, 1], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data: the Concentration of reactant B (CB)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant B (CB)')
plt.legend(['Expected: CB', 'NN: CB'])
plt.show()
# %%

#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(time_x_axis_testing, y_test_HyperParametersModel[0], 'b')
plt.plot(time_x_axis_testing, y_pred_test[:, 0], 'r--')
plt.title('Closed Loop Simulation on PGNN for TESTING data sets: the Concentration of reactant A (CA)')
plt.xlabel('time')
plt.ylabel('h(t)/m')
plt.legend(['Expected: CA', 'NN: CA'])
plt.show()
# %%


#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(time_x_axis_testing, y_test_HyperParametersModel[1], 'b')
plt.plot(time_x_axis_testing, y_pred_test[:, 1], 'r--')
plt.title('Closed Loop Simulation on PGNN for TESTING data: the Concentration of reactant B (CB)')
plt.xlabel('time')
plt.ylabel('h(t)/m')
plt.legend(['Expected: CB', 'NN: CB'])
plt.show()

# %%
concatenate_output_expected = pd.concat(
    [y_train_HyperParametersModel, y_test_HyperParametersModel], axis=0,)
y_pred_train_dataframe = pd.DataFrame(y_pred_train)
y_pred_test_dataframe = pd.DataFrame(y_pred_test)
concatenate_output_predicted = pd.concat(
    [y_pred_train_dataframe, y_pred_test_dataframe], axis=0)

# %%
time_x_axis_combined = np.linspace(
    0, int(3600 - input_delay_size), int(3600 - input_delay_size))

# %%
print('the Concentration of reactant A (CA): Combined')
#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(time_x_axis_combined, concatenate_output_expected[0], 'b')
plt.plot(time_x_axis_combined, concatenate_output_predicted[0], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data sets: the Concentration of reactant A (CA)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant A (CA)')
plt.legend(['Expected: CA', 'NN: CA'])
plt.show()

# %%
print('the Concentration of reactant B (CB)')

#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(time_x_axis_combined, concatenate_output_expected[1], 'b')
plt.plot(time_x_axis_combined, concatenate_output_predicted[1], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data: the Concentration of reactant B (CB)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant B (CB)')
plt.legend(['Expected: CB', 'NN: CB'])
plt.show()

# %%
# %%
print('Training and Validation Loss')
loss_on_training_set = history.history['loss']
loss_on_testing_set = history.history['val_loss']
x_axis = range(0, epochs_for_the_best_Hyperparameters)

plt.plot(x_axis, loss_on_training_set, 'g', label='Loss on Training Set')
plt.plot(x_axis, loss_on_testing_set, 'b', label='Loss on Testing Set')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
# %%
