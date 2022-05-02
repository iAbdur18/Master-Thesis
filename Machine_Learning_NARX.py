# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Mar 18 09:23:02 2022

@author: Abdur Rehman
"""

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
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
print('Importing the libraries')
# Importing the libraries
#from Continuous_Stirred_Tank_Reactor import nU_delay, nY_delay
#from tensorflow.keras.metrics.RootMeanSquaredError import RootMeanSquaredError

# =============================================================================
# =============================================================================
# =============================================================================

# %%
print('Importing from the differenet functions ')
from Continuous_Stirred_Tank_Reactor import concatenate_array_output, simulated_input_3d, df_max_scaled_input, df_max_scaled_output

# %%
print('Importing Custom Inputs and Outputs Function Being Called')
from NARX_Concatenation_Function_Updated import concateDataForNARX
from Custom_Inputs_and_Outputs_Function import custom_inputs_outputs
# %%
# Defining all of the Callbacks
print('Setting all of the Callbacks')

callback_EarlyStopping = EarlyStopping(
    monitor="mean_squared_error", patience=10, verbose=1)

#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# CSVLogger logs epoch, Loss etc
callback_CSV_File = CSVLogger(
    'NARX_CSV_Logs_20.csv', separator=',', append=False)

# Terminate On NaN
callback_Terminate_On_NaN = tf.keras.callbacks.TerminateOnNaN()

#
callback_TensorBoard = TensorBoard("/tmp/tb_logs_20")

callback_RemoteMonitor = tf.keras.callbacks.RemoteMonitor(
    root='http://localhost:9000',
    path='/publish/epoch/end/',
    field='data',
    headers=None,
    send_as_json=False
)

callback_list = [callback_EarlyStopping,
                 callback_CSV_File, callback_TensorBoard]

# %%    KerasTuner: scalable hyperparameter optimization framework

# Importing Keras Tuner: Random Search

# Writing a function, which returns a compiled Keras model

# The first thing we need to do is writing a function, which returns a compiled
# Keras model. It takes an argument hp for defining the hyperparameters while
# building the model.
print('KerasTuner: scalable hyperparameter optimization framework')

print('Tune hyperparameters in your custom training loop')


class MyHyperModel(kt.HyperModel):

    def build(self, hp):

        input_delay_size = hp.Int("input_delay_size", 2, 20, step=1)

        input_layers = int((input_delay_size*2)+(input_delay_size * 4))
        #print(input_layers)
        # X_train, X_test, y_train, y_test, input_layers, output_layers = custom_inputs_outputs(
        #     simulated_input_3d, concatenate_array_output, input_delay_size)

        activations = hp.Choice('activation_', ["relu", "tanh", "sigmoid"])

        # groups a linear stack of layers into a tf.keras.Model.
        model = tf.keras.models.Sequential()

        # Defining the Input Layer with the right input shape
        # See without the Flatten
        model.add(layers.Flatten(input_shape=(input_layers,)))
        #model.add(Dense(1, input_shape=(input_layers,)))

        # Defining the hidden layers from 2 to 20 possible layers
        # hyperparameter number 1: Number of hidden Layers
        for i in range(hp.Int('hidden_layers', 2, 25)):

            # Defining the number of neurons from the 32 to 512
            # hyperparameter number 2: number of neurons
            model.add(layers.Dense(units=hp.Int('units_nodes_' + str(i), min_value=32, max_value=256,
                      step=8), activation=activations))

        # Output layers defined with the Linear Regression Model
        model.add(layers.Dense(4, activation='linear'))

        # hyperparameter number 3: Learning Rate
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-1, 1e-2])), loss='mean_squared_error', metrics=[
                      'mean_squared_error', 'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()])

        return model

    def fit(self, hp, model, *args, **kwargs):

        input_delay_size = hp.get("input_delay_size")

        #print(input_delay_size)

        X_train, X_test, y_train, y_test, input_layers, output_layers = custom_inputs_outputs(simulated_input_3d, concatenate_array_output, input_delay_size)

        #print(input_delay_size, input_layers)

        return model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),

            *args,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )


# %%
tuner = kt.RandomSearch(
    MyHyperModel(),
    objective="mean_squared_error",
    max_trials=30,
    overwrite=True,
    executions_per_trial=5,
    directory='project_20',
    project_name="tune_hypermodel",
)
# %%
# Checking the summary

tuner.search_space_summary()

# %%
print('Start of the Hyperparameter Tuning')

tic_Hyperparameter_Tuning = time.perf_counter() # Start Time

tuner.search(epochs=150, verbose=1, callbacks=[callback_list])

toc_Hyperparameter_Tuning = time.perf_counter() # End Time

print(f"Build finished in {(toc_Hyperparameter_Tuning - tic_Hyperparameter_Tuning)/60:0.0f} minutes {(toc_Hyperparameter_Tuning - tic_Hyperparameter_Tuning)%60:0.0f} seconds")
print(f"Build finished in {toc_Hyperparameter_Tuning - tic_Hyperparameter_Tuning:0.4f} seconds")
print('End of the Hyperparameter Tuning')
# %%
tuner.results_summary()

# %%
#best_hps = tuner.get_best_hyperparameters()
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
# %%
# Creating the model based on the Best HyperParameters
history_model = tuner.hypermodel.build(best_hp)
history_model.summary()

# %%
# Get the input_delay_size from the best HyperParameters
best_hp.values.get('input_delay_size')

# %%
print('Manual Integration of the Best fit model')

# Find the best Input Delay Size
input_delay_size_from_the_HyperParametersModel = int(
    best_hp.values.get('input_delay_size'))

X_train_HyperParametersModel, X_test_HyperParametersModel, y_train_HyperParametersModel, y_test_HyperParametersModel, input_layers_HyperParametersModel, output_layers_HyperParametersModel = custom_inputs_outputs(
    simulated_input_3d, concatenate_array_output, input_delay_size_from_the_HyperParametersModel)
# %%
epochs_for_the_best_Hyperparameters = int(250)
# %%
print('creating the fit model ')
history = history_model.fit(X_train_HyperParametersModel, y_train_HyperParametersModel, epochs=epochs_for_the_best_Hyperparameters,
                            validation_data=(X_test_HyperParametersModel, y_test_HyperParametersModel), callbacks=[TensorBoard("/tmp/tb_logs_250")])
# %%
print('Evaluation on the Test Set')
history_eval_dict_test = history_model.evaluate(
    X_test_HyperParametersModel, y_test_HyperParametersModel, return_dict=True)
# %%
print('Evaluation on the Train Set')
history_eval_dict_train = history_model.evaluate(
    X_train_HyperParametersModel, y_train_HyperParametersModel, return_dict=True)

# %%
print('Plotting of the History Model ')
plot_model(history_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# =============================================================================
# %%
# =============================================================================
# =============================================================================
# =============================================================================
print('Plotting the Model based on the Best Hyperparameters Received')

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
print('Training and Validation Mean Absolute Error')
mean_absolute_error_on_training_set = history.history['mean_absolute_error']
mean_absolute_error_on_testing_set = history.history['val_mean_absolute_error']
x_axis = range(0, epochs_for_the_best_Hyperparameters)

plt.plot(x_axis, mean_absolute_error_on_training_set,
         'g', label='Training Set')
plt.plot(x_axis, mean_absolute_error_on_testing_set, 'b', label='Testing Set')
plt.title('Training and Validation Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
# %%
print('Training and Validation Root Mean Squared Error')
root_mean_squared_error_on_training_set = history.history['root_mean_squared_error']
root_mean_squared_error_on_testing_set = history.history['val_root_mean_squared_error']
x_axis = range(0, epochs_for_the_best_Hyperparameters)

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
print('Training and Validation Mean Squared Error')
mean_squared_error_on_training_set = history.history['mean_squared_error']
mean_squared_error_on_testing_set = history.history['val_mean_squared_error']
x_axis = range(0, epochs_for_the_best_Hyperparameters)

plt.plot(x_axis, mean_squared_error_on_training_set, 'g', label='Training Set')
plt.plot(x_axis, mean_squared_error_on_testing_set, 'b', label='Testing Set')
plt.title('Training and Validation Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# %%

print('Plotting of the Relations Between Measured and Model Trained Outputs')


# %%
print('Importing the Libraries and Setting Up the Plotting The Parameters')
from matplotlib import gridspec

row_plot, column_plot = X_train_HyperParametersModel.shape
iteration_to_plot = int(column_plot/6)
output_plot_start_point = iteration_to_plot * 4
epochs_x_axis = np.linspace(3000, 3100,101)

model_data = history_model.predict(X_train_HyperParametersModel.loc[0:100])
model_data = pd.DataFrame(model_data)

model_data_testing = history_model.predict(X_test_HyperParametersModel.loc[3000:3100])
model_data_testing = pd.DataFrame(model_data_testing)

# %%

#1
print('Input on TRAINING Data for the first 100 data points: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
fig_training_temperatures.set_figheight(15)
fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle('Input on TRAINING Data for the first 100 data points: TR and TK respectivley')


axes[0].plot(model_data.loc[0:100, [2, 3]], label=["TR", "TK"])
axes[0].set_ylabel('Data From The Model: TR and TK ')
axes[0].legend(loc='upper left')

axes[1].plot(X_train_HyperParametersModel.loc[0:100, [
         output_plot_start_point-2, output_plot_start_point-1]], label=["TR", "TK"])
axes[1].set_ylabel('Simulated Data: TR and TK ')
axes[1].legend(loc='upper left')

axes[2].plot(X_train_HyperParametersModel.loc[0:100, [
         int(column_plot-2), int(column_plot-1)]], label=["TR", "TK"])
axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Number of Epochs on Shared X Axis ')
axes[2].legend(loc='upper left')

# %%

#2
print('Input on TESTING Data for the first 100 data points: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_testing_temperatures, axes_testing = plt.subplots(3, sharex=True, )
fig_testing_temperatures.set_figheight(15)
fig_testing_temperatures.set_figwidth(15)
fig_testing_temperatures.suptitle('Input on TESTING Data for the first 100 data points: TR and TK respectivley')


axes_testing[0].plot(epochs_x_axis, model_data_testing.loc[0:100, [2, 3]], label=["TR", "TK"])
axes_testing[0].set_ylabel('Data From The Model: TR and TK')
axes_testing[0].legend(loc='upper left')

axes_testing[1].plot(X_test_HyperParametersModel.loc[3000:3100, [
          output_plot_start_point-2, output_plot_start_point-1]], label=["TR", "TK"])
axes_testing[1].set_ylabel('Simulated Data: TR and TK ')
axes_testing[1].legend(loc='upper left')


axes_testing[2].plot(X_test_HyperParametersModel.loc[3000:3100, [
          int(column_plot-2), int(column_plot-1)]], label=["TR", "TK"])
axes_testing[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes_testing[2].set_xlabel('Number of Epochs on Shared X Axis ')
axes_testing[2].legend(loc='upper left')


# %%

#3
print('Input on TRAINING Data for the first 100 data points: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_concentrations, axes = plt.subplots(3, sharex=True)
fig_training_concentrations.set_figheight(15)
fig_training_concentrations.set_figwidth(15)
fig_training_concentrations.suptitle('Input on TRAINING Data for the first 100 data points: CA and CB respectivley')


axes[0].plot(model_data.loc[0:100, [0, 1]], label=["CA", "CB"])
axes[0].set_ylabel('Data From The Model: CA and CB ')
axes[0].legend(loc='upper left')

axes[1].plot(X_train_HyperParametersModel.loc[0:100, [
         output_plot_start_point-4, output_plot_start_point-3]], label=["CA", "CB"])
axes[1].set_ylabel('Simulated Data: CA and CB ')
axes[1].legend(loc='upper left')

axes[2].plot(X_train_HyperParametersModel.loc[0:100, [
         int(column_plot-2), int(column_plot-1)]], label=["CA", "CB"])
axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Number of Epochs on Shared X Axis ')
axes[2].legend(loc='upper left')



# %%

#4
print('Input on TESTING Data for the first 100 data points: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_testing_concentrations, axes_testing = plt.subplots(3, sharex=True, )
fig_testing_concentrations.set_figheight(15)
fig_testing_concentrations.set_figwidth(15)
fig_testing_concentrations.suptitle('Input on TESTING Data for the first 100 data points: CA and CB respectivley')


axes_testing[0].plot(epochs_x_axis, model_data_testing.loc[0:100, [0, 1]], label=["CA", "CB"])
axes_testing[0].set_ylabel('Data From The Model: CA and CB ')
axes_testing[0].legend(loc='upper left')

axes_testing[1].plot(X_test_HyperParametersModel.loc[3000:3100, [
          output_plot_start_point-4, output_plot_start_point-3]], label=["CA", "CB"])
axes_testing[1].set_ylabel('Simulated Data: CA and CB ')
axes_testing[1].legend(loc='upper left')


axes_testing[2].plot(X_test_HyperParametersModel.loc[3000:3100, [
          int(column_plot-2), int(column_plot-1)]], label=["CA", "CB"])
axes_testing[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes_testing[2].set_xlabel('Number of Epochs on Shared X Axis ')
axes_testing[2].legend(loc='upper left')


# =============================================================================
# =============================================================================
# =============================================================================



# %%
# =============================================================================
# =============================================================================
# =============================================================================

# %%
#start = time.process_time()
print('Starting Time:')
tic_closed_loop = time.perf_counter() # Start Time

# Closed Loop Simulation
print('Closed Loop Simulation Started')
# Needs to make sure the model and the inputs are built on the same parameters
inputs = X_train_HyperParametersModel

# Finding out the rows and columns for the closed loop simulation array
row_closed_loop, column_closed_loop = inputs.shape

# Creation of the Closed Loop array
#closed_loop_NParray = np.zeros((row_closed_loop+1, column_closed_loop))
closed_loop_NParray = np.zeros((row_closed_loop, column_closed_loop))

# Setting up the initail Valuse for the closed loop simulation from the inputs[0] to dummy variable [0]
dummy_0 = inputs.loc[0]
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
    tic_row_iteration = time.perf_counter() # Start Time
    
    
    # Column Iteration
    for column_iteration in range(column_closed_loop):
        tic_column_iteration = time.perf_counter() # Start Time
        
           
        # Taking the entire row and feeding it to the Neural Network (X_K)
        states_from_the_model = closed_loop_NParray[row_iteration]
        states_from_the_model = numpy.reshape(states_from_the_model, (1, column_closed_loop))
        
        tic_Row_Feeding = time.perf_counter() # Start Time 
        states_from_the_model = history_model.predict(states_from_the_model)
        
        toc_Row_Feeding = time.perf_counter() # End Time
        print(f"Build finished in ROW FEEDING {(toc_Row_Feeding - tic_Row_Feeding)/60:0.0f} minutes {(toc_Row_Feeding - tic_Row_Feeding)%60:0.0f} seconds")
        print(f"Build finished in ROW FEEDING {toc_Row_Feeding - tic_Row_Feeding:0.4f} seconds")
        
        # Shifting all of the STATES based on the delay (X_K-1)
       
        tic_STATES_Shifting = time.perf_counter() # Start Time
        
        for iteration_var in range(1, iteration):
            #print(row_iteration, iteration_var,)
            #print(row_iteration+1, 4*iteration_var, 4 * iteration_var + 4,
            #      row_iteration, 4*iteration_var-4, 4*iteration_var)
            closed_loop_NParray[row_iteration+1, 4*iteration_var:4 *iteration_var + 4] = closed_loop_NParray[row_iteration, 4*iteration_var-4:4*iteration_var]
            iteration_var += 1
        
        toc_STATES_Shifting = time.perf_counter() # End Time
        print(f"Build finished in STATES_Shifting {(toc_STATES_Shifting - tic_STATES_Shifting)/60:0.0f} minutes {(toc_STATES_Shifting - tic_STATES_Shifting)%60:0.0f} seconds")
        print(f"Build finished in STATES_Shifting  {toc_STATES_Shifting - tic_STATES_Shifting:0.4f} seconds")

        closed_loop_NParray[row_iteration+1, 0:4] = states_from_the_model

        # Shifting all of the Inputs based on the delay (U_k-1)
        tic_Inputs_Shifting = time.perf_counter() # Start Time
        
        for iteration_var in range(0, iteration-1):
            
            # print(row_iteration+1, (input_start+2*iteration_var)+2, (input_start+2*iteration_var)+4,
            #       row_iteration, input_start+2*iteration_var, (input_start+2*iteration_var)+2)

            closed_loop_NParray[row_iteration+1, (input_start+2*iteration_var)+2:(input_start+2*iteration_var) + 4] = closed_loop_NParray[row_iteration, input_start+2*iteration_var:(input_start+2*iteration_var)+2]

            iteration_var += 1
        toc_Inputs_Shifting = time.perf_counter() # End Time
        print(f"Build finished in Inputs_Shifting {(toc_Inputs_Shifting - tic_Inputs_Shifting)/60:0.0f} minutes {(toc_Inputs_Shifting - tic_Inputs_Shifting)%60:0.0f} seconds")
        print(f"Build finished in Inputs_Shifting {toc_Inputs_Shifting - tic_Inputs_Shifting:0.4f} seconds")
        
        # Importing the Inputs to the Closed Loop Simulation (U_k)
        closed_loop_NParray[row_iteration+1, input_start:input_start + 2] = inputs.loc[row_iteration+1, input_start:input_start+1]
        
        
        column_iteration += 1
        
        toc_column_iteration = time.perf_counter() # End Time
        print(f"Build finished in column_iteration  {(toc_column_iteration - tic_column_iteration)/60:0.0f} minutes {(toc_column_iteration - tic_column_iteration)%60:0.0f} seconds")
        print(f"Build finished in column_iteration  {toc_column_iteration - tic_column_iteration:0.4f} seconds")

    row_iteration += 1
    
    toc_row_iteration = time.perf_counter() # End Time
    print(f"Build finished in row_iteration  {(toc_row_iteration - tic_row_iteration)/60:0.0f} minutes {(toc_row_iteration - tic_row_iteration)%60:0.0f} seconds")
    print(f"Build finished in row_iteration  {toc_row_iteration - tic_row_iteration:0.4f} seconds")


toc_closed_loop = time.perf_counter() # End Time
print(f"Build finished in  closed_loop {(toc_closed_loop - tic_closed_loop)/60:0.0f} minutes {(toc_closed_loop - tic_closed_loop)%60:0.0f} seconds")
print(f"Build finished in  closed_loop {toc_closed_loop - tic_closed_loop:0.4f} seconds")
print('Closed Loop Simulation Ended')
# print('Fininshing time: ')
# print(time.process_time() - start)

# %%
