# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 19:05:35 2022

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
print('Importing Custom Inputs (X_train and X_test ) and Outputs (y_train and y_test) Function Being Called')
X_train_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/X_train_HyperParametersModel.xlsx")
X_test_HyperParametersModel = pd.read_excel("C:/Users/Abdur Rehman/Thesis Code/X_test_HyperParametersModel.xlsx")
y_train_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/y_train_HyperParametersModel.xlsx")
y_test_HyperParametersModel = pd.read_excel("C:/Users/Abdur Rehman/Thesis Code/y_test_HyperParametersModel.xlsx")

# %%
print(' Importing the Best Hyperparameters Values from the Vanialla Netwrok ')

best_hp_Dataframe = pd.read_excel("C:/Users/Abdur Rehman/Thesis Code/best_hp_Dataframe.xlsx")

# %%
print(' Extracting the values from the Best Hyperparameters variable')

input_delay_size = best_hp_Dataframe['input_delay_size']
activation_ = best_hp_Dataframe['activation_']
learning_rate = best_hp_Dataframe['learning_rate']
units_nodes_ = best_hp_Dataframe['units_nodes_']
hidden_layers = best_hp_Dataframe['hidden_layers']
momentum_best = best_hp_Dataframe['momentum']
batch_size_best = best_hp_Dataframe['batch_size']


# %%
input_layers = int((input_delay_size*2)+(input_delay_size * 4))
# %%
# groups a linear stack of layers into a tf.keras.Model.
model = tf.keras.models.Sequential()

# Defining the Input Layer with the right input shape
model.add(layers.Flatten(input_shape=(int(input_layers),)))

# Defining the hidden layers
# input_shape=(input_layers,),
model.add(layers.Dense(units= int(units_nodes_[0]), activation='relu',
                # kernel_regularizer=regularizers.L1(l1=1e-3),
                bias_regularizer=regularizers.L2(1e-3),
                # activity_regularizer=regularizers.L1(1e-3)
                ))
model.add(layers.Dense(units=int(units_nodes_[0]), activation='relu',
                # kernel_regularizer=regularizers.L1(l1=1e-3),
                bias_regularizer=regularizers.L2(1e-3),
                # activity_regularizer=regularizers.L1(1e-3)
                ))
model.add(layers.Dense(units=int(units_nodes_[0]), activation='relu',
                # kernel_regularizer=regularizers.L1(l1=1e-3),
                bias_regularizer=regularizers.L2(1e-3),
                # activity_regularizer=regularizers.L1(1e-3)
                ))

#Output layer
model.add(layers.Dense(4, activation='linear'))

# %%
# compile the keras model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = learning_rate[0], momentum = momentum_best[0], name="SGD"), loss='mean_squared_error', metrics=[
              'mean_squared_error', 'accuracy', tf.keras.metrics.RootMeanSquaredError()])
# %%
loss_to_be_calculated=[]
epochs_number = int(10000)
# number_to_be_averaged = int(3)

# for i in range(number_to_be_averaged):
    
history = model.fit(X_train_HyperParametersModel, y_train_HyperParametersModel,
                    validation_data=(X_test_HyperParametersModel, y_test_HyperParametersModel),
                    epochs=epochs_number, batch_size=int(batch_size_best[0]),)

# %%
# evaluate the keras model
print('accuracy_train')
accuracy_train = model.evaluate(X_train_HyperParametersModel, y_train_HyperParametersModel)
# %%
print('accuracy_test')
accuracy_test = model.evaluate(X_test_HyperParametersModel, y_test_HyperParametersModel)
# %%
model.summary()
# %%
print('Plotting of the History Model ')
plot_model(model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)
# =============================================================================
# %%
print('Training and Validation Loss')
loss_on_training_set_BHPS = history.history['loss']
loss_on_testing_set_BHPS = history.history['val_loss']
x_axis = range(0, epochs_number)
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(x_axis, loss_on_training_set_BHPS, 'g', label='Loss on Training Set')
plt.plot(x_axis, loss_on_testing_set_BHPS, 'b', label='Loss on Testing Set')
plt.title('Training and Validation Loss ')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
print('Training and Validation Mean Absolute Error')
accuracy_training_set = history.history['accuracy']
accuracy_testing_set = history.history['val_accuracy']
x_axis = range(0, epochs_number)

plt.plot(x_axis, mean_absolute_error_on_training_set,
         'g', label='Training Set')
plt.plot(x_axis, mean_absolute_error_on_testing_set, 'b', label='Testing Set')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# %%
print('Training and Validation Root Mean Squared Error')
root_mean_squared_error_on_training_set = history.history['root_mean_squared_error']
root_mean_squared_error_on_testing_set = history.history['val_root_mean_squared_error']
x_axis = range(0, epochs_number)

plt.plot(x_axis, root_mean_squared_error_on_training_set,
         'g', label='Training Set')
plt.plot(x_axis, root_mean_squared_error_on_testing_set,
         'b', label='Testing Set')
plt.title('Training and Validation Root Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('Root Mean Squared Error')
plt.legend()
plt.show()
# %%
# make probability predictions with the model
predictions_y_train_HyperParametersModel = model.predict(X_train_HyperParametersModel)
predictions_y_train_HyperParametersModel = pd.DataFrame(predictions_y_train_HyperParametersModel)

predictions_y_test_HyperParametersModel = model.predict(X_test_HyperParametersModel)
predictions_y_test_HyperParametersModel = pd.DataFrame(predictions_y_test_HyperParametersModel)
# %%
time_x_axis_training = np.linspace(0, 3000, 3000)
iteration_to_end = 3600-input_delay_size[0]
time_x_axis_testing = np.linspace(3000, iteration_to_end, iteration_to_end-3000)
row_plot, column_plot = X_train_HyperParametersModel.shape
# %%

# 1
print('Input on TRAINING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = (25, 9)
fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(28)
# fig_training_temperatures.set_figwidth(50)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: Temperature of Reactor TR and Temperature of Cooling Jacket TK respectivley')


axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
             2], color='red', linestyle='dashed', label=["Neural Network: TR"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             2], color='blue', label=["Simulated: TR"])
axes[0].set_ylabel(' Temperature of Reactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
             3], color='red', linestyle='dashed', label=["Neural Network: TK"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             3], color='blue', label=["Simulated: TK"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper right')


axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper right')


# %%
# 2
print('Input on TRAINING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(15)
# fig_training_temperatures.set_figwidth(15)

fig_training_temperatures.suptitle(
    'Input on TRAINING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')


axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
             0], color='red', linestyle='dashed', label=["Neural Network: CA"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             0], color='blue', label=["Simulated: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA) ')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
             1], color='red', linestyle='dashed', label=["Neural Network: CB"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             1], color='blue', label=["Simulated: CB"])
axes[1].set_ylabel(' Concentration of Reactant B (CB) ')
axes[1].legend(loc='upper right')


axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper right')

# %%

# 3
print('Input on TESTING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)

fig_training_temperatures.suptitle(
    'Input on TESTING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')


axes[0].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
             3000, 2], color='red', linestyle='dashed', label=["Neural Network: TR"])
axes[0].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
             2], color='blue', label=["Simulated: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
              3000, 3], color='red', linestyle='dashed', label=["Neural Network: TK"])
axes[1].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
              3], color='blue', label=["Simulated: TK"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper right')

axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])


# %%

# 4
print('Input on TESTING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)

fig_training_temperatures.suptitle(
    'Input on TESTING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

axes[0].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
             3000, 0], color='red', linestyle='dashed', label=["Neural Network: CA"])
axes[0].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
             0], color='blue', label=["Simulated: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA)')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
             3000, 1], color='red', linestyle='dashed', label=["Neural Network: CB"])
axes[1].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
             1], color='blue', label=["Simulated: CB"])
axes[1].set_ylabel(' Concentration of Reactant B (CB) ')
axes[1].legend(loc='upper right')

axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

# %%
#start = time.process_time()

#
tic_closed_loop = time.perf_counter()  # Start Time
print('Starting Time:', tic_closed_loop)
# Closed Loop Simulation
print('Closed Loop Simulation Started on Training Set')
# Needs to make sure the model and the inputs are built on the same parameters
inputs = X_train_HyperParametersModel

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
        states_from_the_model = model.predict(states_from_the_model)

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


# %%
closed_loop_NParray_dataframe_training = pd.DataFrame(closed_loop_NParray[:, [0, 1, 2, 3]])
# %%
#start = time.process_time()

#
tic_closed_loop = time.perf_counter()  # Start Time
print('Starting Time:', tic_closed_loop)
# Closed Loop Simulation
print('Closed Loop Simulation Started on TESTING Set')
# Needs to make sure the model and the inputs are built on the same parameters
inputs = X_test_HyperParametersModel

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
        states_from_the_model = model.predict(states_from_the_model)

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



# %%
closed_loop_NParray_dataframe_testing = pd.DataFrame(closed_loop_NParray[:, [0, 1, 2, 3]])

# %%

print('Closed Loop Simulations Graphs')

# 1
print('Input on TRAINING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(15)
# fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')


# axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
#               2], color='red', linestyle='dashed', label=["NN: TR"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             2], color='blue', label=["Simulated: TR"])
axes[0].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             2], color='green', label=["Closed Loop: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper left')


# axes[1].plot(time_x_axis_training, model_data_training.loc[0:3000,
#              3], color='red', linestyle='dashed', label=["NN: TK"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             3], color='blue', label=["Simulated: TK"])
axes[1].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
            3], color='green', label=["Closed Loop: TK"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper left')


axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper left')


# %%

# 2
print('Input on TRAINING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(15)
# fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')


# axes[0].plot(time_x_axis_training, model_data_training.loc[0:3000,
#              0], color='red', linestyle='dashed', label=["NN: CA"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             0], color='blue', label=["Simulated: CA"])
axes[0].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             0], color='green', label=["Closed Loop: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA) ')
axes[0].legend(loc='upper left')


# axes[1].plot(time_x_axis_training, model_data_training.loc[0:3000,
#              1], color='red', linestyle='dashed', label=["NN: CB"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             1], color='blue', label=["Simulated: CB"])
axes[1].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             1], color='green', label=["Closed Loop: CA"])
axes[1].set_ylabel(' Concentration of Reactant B (CB) ')
axes[1].legend(loc='upper left')

axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper left')

# %%

print('Combine Open and Closed Loop Simulations Graphs')

# 1
print('Input on TRAINING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(15)
# fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')


axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
              2], color='red', linestyle='dashed', label=["Neural Network: TR"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             2], color='blue', label=["Simulated: TR"])
axes[0].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             2], color='green', label=["Closed Loop: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper left')


axes[1].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
              3], color='red', linestyle='dashed', label=["Neural Network: TK"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             3], color='blue', label=["Simulated: TK"])
axes[1].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
            3], color='green', label=["Closed Loop: TK"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper left')


axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper left')


# %%

# 2
print('Input on TRAINING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(15)
# fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')


axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
              0], color='red', linestyle='dashed', label=["Neural Network: CA"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             0], color='blue', label=["Simulated: CA"])
axes[0].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             0], color='green', label=["Closed Loop: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA) ')
axes[0].legend(loc='upper left')


axes[1].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
              1], color='red', linestyle='dashed', label=["Neural Network: CB"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             1], color='blue', label=["Simulated: CB"])
axes[1].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             1], color='green', label=["Closed Loop: CA"])
axes[1].set_ylabel(' Concentration of Reactant B (CB) ')
axes[1].legend(loc='upper left')

axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper left')
