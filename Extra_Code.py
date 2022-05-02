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


