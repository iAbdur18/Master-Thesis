# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:44:54 2022

@author: Abdur Rehman
"""

# %%
# %%
# Defining all of the Callbacks


callback_EarlyStopping = EarlyStopping(
    monitor="mean_absolute_error", patience=5, verbose=1)

#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# CSVLogger logs epoch, Loss etc
callback_CSV_File = CSVLogger(
    'NARX_CSV_Logs_Experiment1.csv', separator=',', append=False)

#
callback_TensorBoard = TensorBoard("/tmp/tb_logs_Experiment1")

callback_list = [callback_EarlyStopping,
                 callback_CSV_File, callback_TensorBoard]
# %%

tuner.search(X_train, y_train, epochs=350, verbose=1,
             validation_data=(X_test, y_test), callbacks=[callback_list])
# Try and define CALLBACKS
# %%
tuner.results_summary()

# %%
#best_hps = tuner.get_best_hyperparameters()
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
history_model = tuner.hypermodel.build(best_hp)
history_model.summary()
# %%
history = history_model.fit(X_train, y_train, epochs=250,
                            validation_data=(X_test, y_test))
# %%
history_eval_dict_test = history_model.evaluate(
    X_test, y_test, return_dict=True)
# %%
history_eval_dict_train = history_model.evaluate(
    X_train, y_train, return_dict=True)


# %%
# %%
def build_model(hp):

    activations = hp.Choice('activation_', ["relu", "tanh", "sigmoid"])

    # groups a linear stack of layers into a tf.keras.Model.
    model = tf.keras.models.Sequential()

    # Defining the Input Layer with the right input shape
    # See without the Flatten
    model.add(layers.Flatten(input_shape=(input_layers,)))
    #model.add(Dense(1, input_shape=(input_layers,)))

    # Defining the hidden layers from 2 to 20 possible layers
    # hyperparameter number 1: Number of hidden Layers
    for i in range(hp.Int('hidden_layers', 2, 20)):

        # Defining the number of neurons from the 32 to 512
        # hyperparameter number 2: number of neurons
        model.add(layers.Dense(units=hp.Int('units_nodes_' + str(i), min_value=32, max_value=128,
                  step=8), activation=activations))

    # Output layers defined with the Linear Regression Model
    model.add(layers.Dense(4, activation='linear'))

    # hyperparameter number 3: Learning Rate
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-1, 1e-2])), loss='mean_absolute_error', metrics=[
                  'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', tf.keras.metrics.RootMeanSquaredError()])
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(hp.Float('learning_rate',
    #                                                             min_value=1e-4, max_value=1e-2, sampling="log")),
    #              loss='mean_absolute_error', metrics=['mean_squared_error',
    #                                                   'mean_absolute_error', 'mean_absolute_percentage_error',
    #                                                  'cosine_proximity'])
    # (hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))
    # model.compile(tf.keras.optimizers.RMSprop(learning_rate=0.1), loss='mean_absolute_error', metrics=[
    #              'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()])

    return model


build_model(kt.HyperParameters())

# =============================================================================
# %%
tuner = RandomSearch(
    build_model,
    objective='mean_absolute_error',
    max_trials=50,
    executions_per_trial=5,
    directory='project_Experiment1',
    project_name='States of the Continuous Stirred Tank Reactor')

# overwrite=True,
# %%
# Checking the summary

tuner.search_space_summary()







# %%
plt.plot(history.history['mean_absolute_error'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['mean_absolute_error', ''], loc='upper left')
plt.show()

# plt.plot(history.history['cosine_proximity'])
# plt.title('Model Accuracy')
# plt.ylabel('cosine_proximity')
# plt.xlabel('Epoch')
# plt.legend(['cosine_proximity', ''], loc='upper left')
# plt.show()

plt.plot(history.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', ''], loc='upper left')
plt.show()

plt.plot(history.history['mean_absolute_percentage_error'])
plt.title('Model Accuracy')
plt.ylabel('mean_absolute_percentage_error')
plt.xlabel('Epoch')
plt.legend(['mean_absolute_percentage_error', ''], loc='upper left')
plt.show()

plt.plot(history.history['mean_squared_error'])
plt.title('Model Accuracy')
plt.ylabel('mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['mean_squared_error', ''], loc='upper left')
plt.show()

plt.plot(history.history['root_mean_squared_error'])
plt.title('Model Accuracy')
plt.ylabel('root_mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['root_mean_squared_error', ''], loc='upper left')
plt.show()

# %%
plt.subplot(1, 2, 1)
plt.plot(X_test[0:50])
plt.ylabel('X_test')
plt.subplot(1, 2, 2)
plt.plot(X_train[:50])
plt.ylabel('X_train')


# %%
fig = plt.Figure(figsize=(20, 3))
gs = gridspec.GridSpec(3, 1, width_ratios=[20,], height_ratios=[10,20,10])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax1.plot(X_train_HyperParametersModel.loc[0:100, [
         42, 43]], label=["TR", "TK"])
plt.show()

# %%
print('Input on Testing Data for the first 100 data points: Feed and Heat Flow')
plt.rcParams["figure.figsize"] = [10, 4]
plt.rcParams["figure.autolayout"] = True
plt.title('Input on Testing Data for the first 100 data points')
plt.ylabel('Feed and Heat Flow ')
plt.plot(X_train_HyperParametersModel.loc[0:100, [
         64, 65]], label=["F", "Q_DOT"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Input on Testing Data for the first 100 data points: Feed and Heat Flow')
plt.rcParams["figure.figsize"] = [10, 4]
plt.rcParams["figure.autolayout"] = True
plt.title('Input on Testing Data for the first 100 data points')
plt.ylabel('Feed and Heat Flow ')
plt.plot(df_max_scaled_input.loc[0:100, [0, 1]], label=["F", "Q_DOT"])
plt.legend(ncol=2, loc="upper right")
# %%
sample_plot = history_model.predict(X_train_HyperParametersModel.loc[0:100])
sample_plot = pd.DataFrame(sample_plot)
print('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.ylabel('temperature TR and the temperature TK ')
plt.plot(sample_plot.loc[0:100, [2, 3]], label=["TR", "TK"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.ylabel('temperature TR and the temperature TK ')
plt.plot(X_train_HyperParametersModel.loc[0:100, [
         42, 43]], label=["TR", "TK"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.ylabel('temperature TR and the temperature TK ')
plt.plot(df_max_scaled_output.loc[0:100, [
         2, 3]], label=["TR", "TK"])
plt.legend(ncol=2, loc="upper right")
# %%
# =============================================================================
# =============================================================================
# =============================================================================
# %%
print('Input on Training Data for the first 100 data points: Feed and Heat Flow')
plt.rcParams["figure.figsize"] = [10, 4]
plt.rcParams["figure.autolayout"] = True
plt.title('Input on Training Data for the first 100 data points')
plt.ylabel('Feed and Heat Flow ')
plt.plot(X_train_HyperParametersModel.loc[0:100, [
         64, 65]], label=["F", "Q_DOT"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Temperature on Training Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Temperature on Training Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.ylabel('temperature TR and the temperature TK ')
plt.plot(X_train_HyperParametersModel.loc[0:100, [2, 3]], label=["TR", "TK"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Temperature on Testing Data: the temperature inside the reactor (TR) and the temperature of the cooling jacket (TK)')
plt.ylabel('temperature TR and the temperature TK ')
plt.plot(X_test_HyperParametersModel.loc[3000:3100, [
         2, 3]], label=["TR", "TK"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Concentration on Training Data: concentration of reactant A (CA), the concentration of reactant B (CB)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Concentration on Training Data: concentration of reactant A (CA), the concentration of reactant B (CB)')
plt.ylabel('Concentration CA and the Concentration CB ')
plt.plot(X_train_HyperParametersModel.loc[0:100, [0, 1]], label=["CA", "CB"])
plt.legend(ncol=2, loc="upper right")
# %%
print('Concentration on Testing Data: concentration of reactant A (CA), the concentration of reactant B (CB)')
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.autolayout"] = True
plt.title('Concentration on Testing Data: concentration of reactant A (CA), the concentration of reactant B (CB)')
plt.ylabel('Concentration CA and the Concentration CB ')
plt.plot(X_test_HyperParametersModel.loc[3000:3100, [
         0, 1]], label=["CA", "CB"])
plt.legend(ncol=2, loc="upper right")


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

# %%
#start = time.process_time()
print('Starting Time:')
# tic_closed_loop = time.perf_counter() # Start Time

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
    # tic_row_iteration = time.perf_counter() # Start Time
    
    
    # Column Iteration
    for column_iteration in range(column_closed_loop):
        # tic_column_iteration = time.perf_counter() # Start Time
        
           
        # Taking the entire row and feeding it to the Neural Network (X_K)
        states_from_the_model = closed_loop_NParray[row_iteration]
        states_from_the_model = numpy.reshape(states_from_the_model, (1, column_closed_loop))
        
        # tic_Row_Feeding = time.perf_counter() # Start Time 
        states_from_the_model = history_model.predict(states_from_the_model)
        
        # toc_Row_Feeding = time.perf_counter() # End Time
        # print(f"Build finished in ROW FEEDING {(toc_Row_Feeding - tic_Row_Feeding)/60:0.0f} minutes {(toc_Row_Feeding - tic_Row_Feeding)%60:0.0f} seconds")
        # print(f"Build finished in ROW FEEDING {toc_Row_Feeding - tic_Row_Feeding:0.4f} seconds")
        
        # Shifting all of the STATES based on the delay (X_K-1)
       
        # tic_STATES_Shifting = time.perf_counter() # Start Time
        
        for iteration_var in range(1, iteration):
            #print(row_iteration, iteration_var,)
            #print(row_iteration+1, 4*iteration_var, 4 * iteration_var + 4,
            #      row_iteration, 4*iteration_var-4, 4*iteration_var)
            closed_loop_NParray[row_iteration+1, 4*iteration_var:4 *iteration_var + 4] = closed_loop_NParray[row_iteration, 4*iteration_var-4:4*iteration_var]
            iteration_var += 1
        
        # toc_STATES_Shifting = time.perf_counter() # End Time
        # print(f"Build finished in STATES_Shifting {(toc_STATES_Shifting - tic_STATES_Shifting)/60:0.0f} minutes {(toc_STATES_Shifting - tic_STATES_Shifting)%60:0.0f} seconds")
        # print(f"Build finished in STATES_Shifting  {toc_STATES_Shifting - tic_STATES_Shifting:0.4f} seconds")

        closed_loop_NParray[row_iteration+1, 0:4] = states_from_the_model

        # Shifting all of the Inputs based on the delay (U_k-1)
        # tic_Inputs_Shifting = time.perf_counter() # Start Time
        
        for iteration_var in range(0, iteration-1):
            
            # print(row_iteration+1, (input_start+2*iteration_var)+2, (input_start+2*iteration_var)+4,
            #       row_iteration, input_start+2*iteration_var, (input_start+2*iteration_var)+2)

            closed_loop_NParray[row_iteration+1, (input_start+2*iteration_var)+2:(input_start+2*iteration_var) + 4] = closed_loop_NParray[row_iteration, input_start+2*iteration_var:(input_start+2*iteration_var)+2]

            iteration_var += 1
        # toc_Inputs_Shifting = time.perf_counter() # End Time
        # print(f"Build finished in Inputs_Shifting {(toc_Inputs_Shifting - tic_Inputs_Shifting)/60:0.0f} minutes {(toc_Inputs_Shifting - tic_Inputs_Shifting)%60:0.0f} seconds")
        # print(f"Build finished in Inputs_Shifting {toc_Inputs_Shifting - tic_Inputs_Shifting:0.4f} seconds")
        
        # Importing the Inputs to the Closed Loop Simulation (U_k)
        closed_loop_NParray[row_iteration+1, input_start:input_start + 2] = inputs.loc[row_iteration+1, input_start:input_start+1]
        
        
        column_iteration += 1
        
        # toc_column_iteration = time.perf_counter() # End Time
        # print(f"Build finished in column_iteration  {(toc_column_iteration - tic_column_iteration)/60:0.0f} minutes {(toc_column_iteration - tic_column_iteration)%60:0.0f} seconds")
        # print(f"Build finished in column_iteration  {toc_column_iteration - tic_column_iteration:0.4f} seconds")

    row_iteration += 1
    
    # toc_row_iteration = time.perf_counter() # End Time
    # print(f"Build finished in row_iteration  {(toc_row_iteration - tic_row_iteration)/60:0.0f} minutes {(toc_row_iteration - tic_row_iteration)%60:0.0f} seconds")
    # print(f"Build finished in row_iteration  {toc_row_iteration - tic_row_iteration:0.4f} seconds")


# toc_closed_loop = time.perf_counter() # End Time
# print(f"Build finished in  closed_loop {(toc_closed_loop - tic_closed_loop)/60:0.0f} minutes {(toc_closed_loop - tic_closed_loop)%60:0.0f} seconds")
# print(f"Build finished in  closed_loop {toc_closed_loop - tic_closed_loop:0.4f} seconds")
# print('Closed Loop Simulation Ended')
# print('Fininshing time: ')
# print(time.process_time() - start)

# %%
X_train_HyperParametersModel, X_test_HyperParametersModel, y_train_HyperParametersModel, y_test_HyperParametersModel, input_layers_HyperParametersModel, output_layers_HyperParametersModel = custom_inputs_outputs(
    simulated_input_3d, concatenate_array_output, input_delay_size_from_the_HyperParametersModel)

# %%
plt.clf()
print('PLOTS')
plt.plot(df_max_scaled_input)
plt.title('The Control Inputs')
plt.ylabel('Inputs')
plt.xlabel('Iterations')
plt.legend(['Feed', 'Heat Flow'], loc='upper left')
plt.show()



# %%
def network_derivative_loss(params_1, params_2):

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
    
    def custom_physics_loss(X_train_HyperParametersModel, y_train_HyperParametersModel, model):
        """
        Here we will use the physics information (differential equation) to steer the network towards making
        correct predictions for unseen / unlabelled data.
        """
        # bring the closed loop model here
        # dont get negative constraints for Ca and CB
        # evaluate this on closed loop predictions

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


