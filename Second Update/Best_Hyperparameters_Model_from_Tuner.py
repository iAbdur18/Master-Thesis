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
                bias_regularizer=regularizers.L1(1e-3),
                activity_regularizer=regularizers.L1(1e-3)
                ))
model.add(layers.Dense(units=int(units_nodes_[0]), activation='relu',
                # kernel_regularizer=regularizers.L1(l1=1e-3),
                bias_regularizer=regularizers.L1(1e-3),
                # activity_regularizer=regularizers.L1(1e-3)
                ))
model.add(layers.Dense(units=int(units_nodes_[0]), activation='relu',
                # kernel_regularizer=regularizers.L1(l1=1e-3),
                bias_regularizer=regularizers.L1(1e-3),
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
epochs_number = int(250)
number_to_be_averaged = int(3)

for i in range(number_to_be_averaged):
    
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
print('Training and Validation Loss')
loss_on_training_set_BHPS = history.history['loss']
loss_on_testing_set_BHPS = history.history['val_loss']
x_axis = range(0, epochs_number)
plt.rcParams["figure.figsize"] = (28, 15)
plt.plot(x_axis, loss_on_training_set_BHPS, 'g', label='Loss on Training Set')
plt.plot(x_axis, loss_on_testing_set_BHPS, 'b', label='Loss on Testing Set')
plt.title('Training and Validation Loss activity_regularizer and bias_regularizer L1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
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

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
# fig_training_temperatures.set_figheight(28)
# fig_training_temperatures.set_figwidth(50)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: TR and TK respectivley')
plt.rcParams["figure.figsize"] = (28, 15)

axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
             2], color='red', linestyle='dashed', label=["NN: TR"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             2], color='blue', label=["Expected: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
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
# fig_training_temperatures.set_figheight(15)
# fig_training_temperatures.set_figwidth(15)
plt.rcParams["figure.figsize"] = (28, 15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: CA and CB respectivley')


axes[0].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
             0], color='red', linestyle='dashed', label=["NN: CA"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             0], color='blue', label=["Expected: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA) ')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_training, predictions_y_train_HyperParametersModel.loc[0:3000,
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

fig_training_temperatures, axes = plt.subplots(2, sharex=True)
plt.rcParams["figure.figsize"] = (28, 15)
fig_training_temperatures.suptitle(
    'Input on TESTING Data: TR and TK respectivley')


axes[0].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
             3000, 2], color='red', linestyle='dashed', label=["NN: TR"])
axes[0].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
             2], color='blue', label=["Expected: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
              3000, 3], color='red', linestyle='dashed', label=["NN: TK"])
axes[1].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
              3], color='blue', label=["Expected: TK"])
axes[1].set_ylabel(' Temperature of Cooling Jacket TK ')
axes[1].legend(loc='upper right')


# %%

# 4
print('Input on TESTING Data: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(2, sharex=True)
plt.rcParams["figure.figsize"] = (28, 15)
fig_training_temperatures.suptitle(
    'Input on TESTING Data: CA and CB respectivley')

axes[0].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
             3000, 0], color='red', linestyle='dashed', label=["NN: CA"])
axes[0].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
             0], color='blue', label=["Expected: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA)')
axes[0].legend(loc='upper right')


axes[1].plot(time_x_axis_testing, predictions_y_test_HyperParametersModel.loc[0:iteration_to_end -
             3000, 1], color='red', linestyle='dashed', label=["NN: CB"])
axes[1].plot(time_x_axis_testing, y_test_HyperParametersModel.loc[0:iteration_to_end-3000,
             1], color='blue', label=["Expected: CB"])
axes[1].set_ylabel(' Concentration of Reactant B (CB) ')
axes[1].legend(loc='upper right')

