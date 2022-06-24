# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 18:22:39 2022

@author: Abdur Rehman
"""

# %%
# Closed Loop Simulation
print('Closed Loop Simulation Started')
# =============================================================================
# =============================================================================
# =============================================================================

# %%
#start = time.process_time()

#
tic_closed_loop = time.perf_counter()  # Start Time
print('Starting Time:', tic_closed_loop)
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


# %%

print('Closed Loop Simulations Graphs')

# %%
# Converting the Closed Loop Numpy to Pandas Dataframe for Training Test Data
#closed_loop_NParray_dataframe_training = closed_loop_function(X_train_HyperParametersModel)
#closed_loop_NParray_dataframe_training = pd.DataFrame(closed_loop_NParray_dataframe_training[:, [0, 1, 2, 3]])
closed_loop_NParray_dataframe_training = pd.DataFrame(closed_loop_NParray[:, [0, 1, 2, 3]])

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
axes[0].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             2], color='green', label=["CL: TR"])
axes[0].set_ylabel(' Temperature of Teactor TR')
axes[0].legend(loc='upper left')


axes[1].plot(time_x_axis_training, model_data_training.loc[0:3000,
             3], color='red', linestyle='dashed', label=["NN: TK"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             3], color='blue', label=["Expected: TK"])
axes[1].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
            3], color='green', label=["CL: TK"])
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
fig_training_temperatures.set_figheight(15)
fig_training_temperatures.set_figwidth(15)
fig_training_temperatures.suptitle(
    'Input on TRAINING Data: CA and CB respectivley')


axes[0].plot(time_x_axis_training, model_data_training.loc[0:3000,
             0], color='red', linestyle='dashed', label=["NN: CA"])
axes[0].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             0], color='blue', label=["Expected: CA"])
axes[0].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             0], color='green', label=["CL: CA"])
axes[0].set_ylabel(' Concentration of Reactant A (CA) ')
axes[0].legend(loc='upper left')


axes[1].plot(time_x_axis_training, model_data_training.loc[0:3000,
             1], color='red', linestyle='dashed', label=["NN: CB"])
axes[1].plot(time_x_axis_training, y_train_HyperParametersModel.loc[0:3000,
             1], color='blue', label=["Expected: CB"])
axes[1].plot(time_x_axis_training, closed_loop_NParray_dataframe_training.loc[0:3000,
             1], color='green', label=["CL: CA"])
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
        
        
