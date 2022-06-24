# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Mar 18 09:23:02 2022

@author: Abdur Rehman
"""

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
    monitor="mean_squared_error", patience=5, verbose=1)

#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# CSVLogger logs epoch, Loss etc
callback_CSV_File = CSVLogger(
    'NARX_CSV_Logs_300_Max_Trials.csv', separator=',', append=False)

# Terminate On NaN
callback_Terminate_On_NaN = tf.keras.callbacks.TerminateOnNaN()

#
callback_TensorBoard = TensorBoard("/tmp/tb_logs_max_trial_100")

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

        activations = hp.Choice('activation_', ["relu", "tanh", "sigmoid"])
        
        units_hp = hp.Int('units_nodes_', min_value=10, max_value=90,
                          step=10)
        
        # groups a linear stack of layers into a tf.keras.Model.
        model = tf.keras.models.Sequential()

        # Defining the Input Layer with the right input shape
        # See without the FlattenS
        model.add(layers.Flatten(input_shape=(input_layers,)))
        #model.add(Dense(1, input_shape=(input_layers,)))

        # Defining the hidden layers from 2 to 20 possible layers
        # hyperparameter number 1: Number of hidden Layers
        for i in range(hp.Int('hidden_layers', 2, 10)):

            # Defining the number of neurons from the 32 to 512
            # hyperparameter number 2: number of neurons

            # Take it out from the loop and just change it once

            model.add(layers.Dense(units=units_hp, activation=activations,
                                   bias_regularizer=regularizers.L2(1e-3)))

        # Output layers defined with the Linear Regression Model
        model.add(layers.Dense(4, activation='linear'))

        # hyperparameter number 3: Learning Rate
        model.compile(optimizer=tf.keras.optimizers.SGD(hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3]), hp.Choice('momentum', [0.3, 0.5, 0.7, 0.9 ]), name="SGD"), loss='mean_squared_error', metrics=[
                      'mean_squared_error', 'accuracy', tf.keras.metrics.RootMeanSquaredError()])

        return model

    def fit(self, hp, model, *args, **kwargs):

        input_delay_size = hp.get("input_delay_size")

        #print(input_delay_size)

        X_train, X_test, y_train, y_test, input_layers, output_layers = custom_inputs_outputs(
            simulated_input_3d, concatenate_array_output, input_delay_size)

        #print(input_delay_size, input_layers)

        return model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=hp.Choice("batch_size", [32, 64, 128, 256]),
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
    directory='30_Max_Trials',
    project_name="tune_hypermodel",
)
# %%
# Checking the summary

tuner.search_space_summary()

# %%
print('Start of the Hyperparameter Tuning')

tic_Hyperparameter_Tuning = time.perf_counter()  # Start Time

tuner.search(epochs=250, verbose=1, callbacks=[callback_list])

toc_Hyperparameter_Tuning = time.perf_counter()  # End Time

print(f"Build finished in {(toc_Hyperparameter_Tuning - tic_Hyperparameter_Tuning)/60:0.0f} minutes {(toc_Hyperparameter_Tuning - tic_Hyperparameter_Tuning)%60:0.0f} seconds")
print(
    f"Build finished in {toc_Hyperparameter_Tuning - tic_Hyperparameter_Tuning:0.4f} seconds")
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
# Storing the Data in a CSV file
print('Storing the Data in a CSV file')
best_hp_Dataframe = pd.DataFrame([best_hp.values])
best_hp_Dataframe.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\best_hp_Dataframe.xlsx', index=False)

# %%
print('Manual Integration of the Best fit model')

# Find the best Input Delay Size
input_delay_size_from_the_HyperParametersModel = int(
    best_hp.values.get('input_delay_size'))

X_train_HyperParametersModel, X_test_HyperParametersModel, y_train_HyperParametersModel, y_test_HyperParametersModel, input_layers_HyperParametersModel, output_layers_HyperParametersModel = custom_inputs_outputs(
    simulated_input_3d, concatenate_array_output, input_delay_size_from_the_HyperParametersModel)
# %%
# Storing the Data in a CSV file
print('Storing the TEST and TRAIN Datasets in separate CSV files')
X_train_HyperParametersModel.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\X_train_HyperParametersModel.xlsx', index=False)
X_test_HyperParametersModel.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\X_test_HyperParametersModel.xlsx', index=False)
y_train_HyperParametersModel.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\y_train_HyperParametersModel.xlsx', index=False)
y_test_HyperParametersModel.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\y_test_HyperParametersModel.xlsx', index=False)


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
plot_model(history_model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)
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

row_plot, column_plot = X_train_HyperParametersModel.shape
iteration_to_plot = int(column_plot/6)
iteration_to_end = 3600-iteration_to_plot

output_plot_start_point = iteration_to_plot * 4
time_x_axis_testing = np.linspace(
    3000, iteration_to_end, iteration_to_end-3000)
time_x_axis_training = np.linspace(0, 3000, 3000)

model_data_training = history_model.predict(
    X_train_HyperParametersModel.loc[0:3000])
model_data_training = pd.DataFrame(model_data_training)

model_data_testing = history_model.predict(
    X_test_HyperParametersModel.loc[3000:iteration_to_end])
model_data_testing = pd.DataFrame(model_data_testing)

# %%

# 1
print('Input on TRAINING Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
fig_training_temperatures.set_figheight(15)
fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: TR and TK respectivley')


axes[0].plot(time_x_axis_training, model_data_training.loc[0:3000,
             2], color='red', linestyle='dashed', label=["NN: TR"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             2], color='blue', label=["Expected: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_training, model_data_training.loc[0:3000,
             3], color='red', linestyle='dashed', label=["NN: TK"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             3], color='blue', label=["Expected: TK"])
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
fig_training_temperatures.set_figheight(15)
fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: CA and CB respectivley')


axes[0].plot(time_x_axis_training, model_data_training.loc[0:3000,
             0], color='red', linestyle='dashed', label=["NN: CA"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             0], color='blue', label=["Expected: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA) ')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_training, model_data_training.loc[0:3000,
             1], color='red', linestyle='dashed', label=["NN: CB"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             1], color='blue', label=["Expected: CB"])
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
fig_training_temperatures.set_figheight(15)
fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TESTING Data: TR and TK respectivley')


axes[0].plot(time_x_axis_testing, model_data_testing.loc[0:iteration_to_end -
             3000, 2], color='red', linestyle='dashed', label=["NN: TR"])
axes[0].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[3000:iteration_to_end,
             2], color='blue', label=["Expected: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_testing, model_data_testing.loc[0:iteration_to_end -
             3000, 3], color='red', linestyle='dashed', label=["NN: TK"])
axes[1].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[3000:iteration_to_end,
             3], color='blue', label=["Expected: TK"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper right')


axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[3000:iteration_to_end, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[3000:iteration_to_end, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper right')

# %%

# 4
print('Input on TESTING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
fig_training_temperatures.set_figheight(15)
fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TESTING Data: CA and CB respectivley')

axes[0].plot(time_x_axis_testing, model_data_testing.loc[0:iteration_to_end -
             3000, 0], color='red', linestyle='dashed', label=["NN: CA"])
axes[0].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[3000:iteration_to_end,
             0], color='blue', label=["Expected: CA"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_testing, model_data_testing.loc[0:iteration_to_end -
             3000, 1], color='red', linestyle='dashed', label=["NN: CB"])
axes[1].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[3000:iteration_to_end,
             1], color='blue', label=["Expected: CB"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper right')


axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[3000:iteration_to_end, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_testing, X_test_HyperParametersModel.loc[3000:iteration_to_end, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper right')


# =============================================================================
# =============================================================================
# =============================================================================






