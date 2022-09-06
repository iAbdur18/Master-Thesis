# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 18:26:07 2022

@author: Abdur Rehman
"""

# %%
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
print('Importing the libraries')

# %%
print('Importing PGNN libraries')
from tf_pgnn import PGNN
# %%
print('Importing Custom Inputs (X_train and X_test ) and Outputs (y_train and y_test) Function Being Called')
X_train_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/X_train_HyperParametersModel.xlsx")
X_test_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/X_test_HyperParametersModel.xlsx")
y_train_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/y_train_HyperParametersModel.xlsx")
y_test_HyperParametersModel = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/y_test_HyperParametersModel.xlsx")

# %%
rows_CLS, columns_CLS = X_test_HyperParametersModel.shape
# %%
print(' Importing the Best Hyperparameters Values from the Vanialla Netwrok ')

best_hp_Dataframe = pd.read_excel(
    "C:/Users/Abdur Rehman/Thesis Code/best_hp_Dataframe.xlsx")

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

def network_derivative_loss(params_1):
    # must be defined with these input arguments - template must be followed for this to work
    # Needs to make sure the model and the inputs are built on the same parameters
    global inputs
    inputs = params_1
    # Finding out the rows and columns for the closed loop simulation array
    row_closed_loop, column_closed_loop = inputs.shape
    global closed_loop_NParray
    # Creation of the Closed Loop array
    closed_loop_NParray = tf.zeros((row_closed_loop, column_closed_loop))

    # Setting up the initail Valuse for the closed loop simulation from the inputs[0] to dummy variable [0]

    closed_loop_NParray = tf.concat([tf.expand_dims(tf.gather_nd(
        inputs, [0]), axis=0, name=None), closed_loop_NParray], 0)
    closed_loop_NParray, delete_row_from_closed_loop_NParray = tf.split(
        closed_loop_NParray, [596, 1], 0)

    # Defining the rows and columns of the closed loop simulation array
    global row_iteration
    row_iteration = 0
    global column_iteration
    column_iteration = 0

    # Fining the delays nU=nY from the NARX Input
    iteration = int(column_closed_loop/6)

    global iteration_var
    iteration_var = 0

    # Finding the start point of the inputs in an array
    input_start = column_closed_loop - iteration * 2

    def custom_physics_loss(X_train_HyperParametersModel, y_train_HyperParametersModel, model):
        """
        Here we will use the physics information (differential equation) to steer the network towards making
        correct predictions for unseen / unlabelled data.
        """
        # Taking the entire row and feeding it to the Neural Network (X_K)
        global closed_loop_NParray
        global inputs
        global states_from_the_model

        global jacket_temperature_difference
        jacket_temperature_difference = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        global q_dot_difference
        q_dot_difference = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        global tensor_closed_loop
        tensor_closed_loop = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        states_from_the_model = inputs[0]
        states_from_the_model = tf.expand_dims(
            states_from_the_model, axis=0)
        states_from_the_model = model.predict(states_from_the_model)

        for row_iteration in range(rows_CLS - 1):
            column_iteration = 0
            while column_iteration < 1:

                # Creating the values from the tensor
                tensor_closed_loop = tensor_closed_loop.write(row_iteration, tf.concat([states_from_the_model, tf.expand_dims(inputs[row_iteration, 0: input_start - 4], axis=0), tf.expand_dims(
                    inputs[row_iteration+1, column_closed_loop-8:column_closed_loop-6], axis=0), tf.expand_dims(inputs[row_iteration, input_start:column_closed_loop-2], axis=0)], axis=1,))
                # print(tensor_closed_loop.size())
                states_from_the_model = model.predict(
                    tensor_closed_loop.read(row_iteration))
                column_iteration += 1

            row_iteration += 1

        concatenated_tensor_closed_loop = tensor_closed_loop.concat()

        # Constraint Number 3
        # Take the Q_dot out from the Closed Loop Array
        q_dot_y_axis = concatenated_tensor_closed_loop[:, 23]

        # Take the Temperature of the Jacket out from the Closed Loop Array
        jacket_temperature_y_axis = concatenated_tensor_closed_loop[:, 3]

        # Backpropagation and forward propagation stepsizes
        step_difference = int(80)
        rows_for_the_loop = int(np.array(q_dot_y_axis.shape))

        for i in range(rows_for_the_loop-step_difference-1):
            jacket_temperature_difference = jacket_temperature_difference.write(
                i, jacket_temperature_y_axis[(i+(1+step_difference))]-jacket_temperature_y_axis[(i+step_difference)])

        jacket_temperature_difference = jacket_temperature_difference.concat()

        # Subtracting the latest 𝑄 ̇  from the previous
        for i in range(rows_for_the_loop-step_difference-1):
            q_dot_difference = q_dot_difference.write(
                i, q_dot_y_axis[(i+step_difference)]-q_dot_y_axis[i])

        q_dot_difference = q_dot_difference.concat()

        # product of Q_dot with change of Temperature of the Jacket
        product_q_dot_and_jacket_temperature = tf.math.multiply(
            q_dot_difference, jacket_temperature_difference)

        ca_PGNN = -concatenated_tensor_closed_loop[:, 0]
        cb_PGNN = -concatenated_tensor_closed_loop[:, 1]

        
        loss_1 = tf.math.reduce_sum(tf.nn.relu(ca_PGNN))
        loss_2 = tf.math.reduce_sum(tf.nn.relu(cb_PGNN))
        loss_3 = tf.math.reduce_sum(
            tf.nn.relu(-product_q_dot_and_jacket_temperature))

        # loss = tf.math.add(loss_1, loss_2, loss_3)
        
        loss = [loss_1, loss_2]
       
        # loss = loss_1

        # predicted closed loop temperatures out
        return loss

    return custom_physics_loss


# %% create a pgnn
xavier = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

# using an MLP
model = PGNN(

    # tf.keras.Sequential(),

    tf.keras.layers.Flatten(
        input_shape=[X_train_HyperParametersModel.shape[1]]),

    tf.keras.layers.Dense(units_nodes_, kernel_initializer=xavier,
                          activation="sigmoid"),

    tf.keras.layers.Dense(units_nodes_, kernel_initializer=xavier,
                          activation="sigmoid"),

    tf.keras.layers.Dense(units_nodes_, kernel_initializer=xavier,
                          activation="sigmoid"),

    tf.keras.layers.Dense(units_nodes_, kernel_initializer=xavier,
                          activation="sigmoid"),


    tf.keras.layers.Dense(4, kernel_initializer=xavier),

    optimizer=tf.keras.optimizers.SGD(
        learning_rate=learning_rate[0], momentum=momentum_best[0]),

    default_loss_function='mse',

    physics_loss_function=network_derivative_loss(tf.constant(
        X_test_HyperParametersModel, dtype=tf.float32)),


    lambda_default=0.5,
    lambda_physics=0.5,

    # early_stop_limit=2e-6
)

# %%
tic_closed_loop = time.perf_counter()  # Start Time
print('Starting Time:', tic_closed_loop)

model.train_model(tf.constant(X_train_HyperParametersModel, dtype=tf.float32), tf.constant(
    y_train_HyperParametersModel, dtype=tf.float32), num_epochs=50,)

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
column_plot = int(24)
# time_x_axis_testing = np.linspace(
#     3000, 3583, 583)
# %%
print('Closed Loop Simulation on PGNN for TRAINING data sets: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
fig_training_temperatures.suptitle(
    'Closed Loop Simulation on PGNN for TRAINING data sets: the Concentration of reactant A (CA) and the Concentration of reactant B (CB)')
plt.rcParams["figure.figsize"] = (25, 15)

axes[0].plot(time_x_axis_training, y_train_HyperParametersModel[0],
             color='b',  label=["Simulated: CA"])
axes[0].plot(time_x_axis_training, y_pred_train[:, 0], color='red',
             linestyle='dashed', label=["Neural Network: CA"])
axes[0].set_ylabel(' Concentration of reactant A (CA)')
axes[0].legend(loc='upper right')

axes[1].plot(time_x_axis_training, y_train_HyperParametersModel[1],
             color='b',  label=["Simulated: CB"])
axes[1].plot(time_x_axis_training, y_pred_train[:, 1], color='red',
             linestyle='dashed', label=["Neural Network: CB"])
axes[1].set_ylabel(' Concentration of reactant B (CB)')
axes[1].legend(loc='upper right')

axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper right')
# %%
print('the Concentration of reactant A (CA)')
plt.plot(time_x_axis_training, y_train_HyperParametersModel[0], 'b')
plt.plot(time_x_axis_training, y_pred_train[:, 0], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data sets: the Concentration of reactant A (CA)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant A (CA)')
plt.legend(['Expected: CA', 'NN: CA'])
plt.show()
# %%
print('Closed Loop Simulation on PGNN for TRAINING data sets: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')

fig_training_temperatures, axes = plt.subplots(3, sharex=True)
fig_training_temperatures.suptitle(
    'Closed Loop Simulation on PGNN for TRAINING data sets: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = (25, 7)

axes[0].plot(time_x_axis_training, y_train_HyperParametersModel[2],
             color='b',  label=["Simulated: TR"])
axes[0].plot(time_x_axis_training, y_pred_train[:, 2], color='red',
             linestyle='dashed', label=["Neural Network: TR"])
axes[0].set_ylabel(' Temperature of Reactor TR')
axes[0].legend(loc='upper right')

axes[1].plot(time_x_axis_training, y_train_HyperParametersModel[3],
             color='b',  label=["Simulated: TK"])
axes[1].plot(time_x_axis_training, y_pred_train[:, 3], color='red',
             linestyle='dashed', label=["Neural Network: TK"])
axes[1].set_ylabel(' Temperature of Reactor TK')
axes[1].legend(loc='upper right')

axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-2)], label=["F"])
axes[2].plot(time_x_axis_training, X_train_HyperParametersModel.loc[0:3000, int(
    column_plot-1)], label=["Q_dot"])

axes[2].set_ylabel('Inputs: Feed and Heat Flow ')
axes[2].set_xlabel('Time ')
axes[2].legend(loc='upper right')
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
plt.rcParams["figure.figsize"] = (25, 10)
plt.plot(time_x_axis_combined, concatenate_output_expected[0], 'b')
plt.plot(time_x_axis_combined, concatenate_output_predicted[0], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data sets: the Concentration of reactant A (CA)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant A (CA)')
plt.legend(['Simulated: CA', 'PGNN: CA'])
plt.show()

# %%
print('the Concentration of reactant B (CB)')

#plt.figure(random.randint(1, 3000))
plt.rcParams["figure.figsize"] = (25, 10)
plt.plot(time_x_axis_combined, concatenate_output_expected[1], 'b')
plt.plot(time_x_axis_combined, concatenate_output_predicted[1], 'r--')
plt.title('Closed Loop Simulation on PGNN for TRAINING data: the Concentration of reactant B (CB)')
plt.xlabel('time')
plt.ylabel('Concentration of reactant B (CB)')
plt.legend(['Simulated: CB', 'PGNN: CB'])
plt.show()

# %%

# =============================================================================
# =============================================================================
# =============================================================================


# %%
# Collecting all of the Data Points for the graphs to construct
input_simulated_comined_HyperParametersModel = pd.concat(
    [X_train_HyperParametersModel, X_test_HyperParametersModel], ignore_index=True)
output_simulated_comined_HyperParametersModel = pd.concat(
    [y_train_HyperParametersModel, y_test_HyperParametersModel], ignore_index=True)

time_x_axis_complete = np.linspace(0, 3596, 3596)
# %%
# 1
print('Input Data: Feed F and heat flow Q_dot respectivley')

plt.rcParams["figure.figsize"] = (25, 10)

plt.plot(time_x_axis_complete,
         input_simulated_comined_HyperParametersModel[22], "green", linestyle='dashed', label="Feed F ")
plt.plot(time_x_axis_complete,
         input_simulated_comined_HyperParametersModel[23], color='blue', linestyle='dashed', label="Heat Flow Q_dot")
plt.title("Input Data: Feed F and heat flow Q_dot respectivley")
plt.xlabel("Time Step")
plt.ylabel("Feed F and Heat Flow Q_dot")
plt.legend(loc="upper right")
plt.show()
# %%
# 2
print('Output Data: Concentrations of reactant A and B (C_A, C_B) and Temperatures (T_R, T_J)')

plt.rcParams["figure.figsize"] = (25, 10)

plt.plot(time_x_axis_complete,
         output_simulated_comined_HyperParametersModel[0], "black",  label="C_A")
plt.plot(time_x_axis_complete,
         output_simulated_comined_HyperParametersModel[1], color='magenta', label="C_B")
plt.plot(time_x_axis_complete,
         output_simulated_comined_HyperParametersModel[2], color='red', label="T_R")
plt.plot(time_x_axis_complete,
         output_simulated_comined_HyperParametersModel[3], color='cyan', label="T_J")

plt.title("Output Data: Concentrations of reactant A and B (C_A, C_B) and Temperatures (T_R, T_J)")
plt.xlabel("Time Step")
plt.ylabel("Output Data of Concentrations: C_A, C_B and Temperatures: T_R, T_J")
plt.legend(loc="upper right")
plt.show()
# %%
fig_training_temperatures, axes = plt.subplots(2, sharex=True)
plt.rcParams["figure.figsize"] = (25, 10)
fig_training_temperatures.suptitle(
    'Input Data: Feed F and heat flow Q_dot respectivley & Output Data: Concentrations of reactant A and B (C_A, C_B) and Temperatures (T_R, T_J)')

axes[0].plot(time_x_axis_complete,
             output_simulated_comined_HyperParametersModel[0], "black",  label="C_A")
axes[0].plot(time_x_axis_complete,
             output_simulated_comined_HyperParametersModel[1], color='magenta', label="C_B")
axes[0].plot(time_x_axis_complete,
             output_simulated_comined_HyperParametersModel[2], color='red', label="T_R")
axes[0].plot(time_x_axis_complete,
             output_simulated_comined_HyperParametersModel[3], color='cyan', label="T_J")
axes[0].set_ylabel(' Output Data')
axes[0].legend(loc='upper right')


axes[1].plot(input_simulated_comined_HyperParametersModel[22],
             color='green', linestyle='dashed', label=["Feed: F"])
axes[1].plot(input_simulated_comined_HyperParametersModel[23],
             color='blue', linestyle='dashed', label=["Heat Flow: Q_dot"])
axes[1].set_ylabel(' Input Data')
axes[1].legend(loc='upper right')
