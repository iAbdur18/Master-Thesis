# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:41:35 2022

@author: Abdur Rehman
"""

import matplotlib
matplotlib.use('Qt5Agg')
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import os
print(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt

# custom imports
# %%
from tf_pgnn import PGNN
# %%
from helper_functions import  get_working_directory, create_results_folder, create_logger
# %% basic variable definitions

delta_x = 0.01
simulation_time = 1
x = np.arange(0, simulation_time + delta_x, delta_x).reshape(-1, 1)  # Numerical grid, time vector

u = np.sin(2 * math.pi * x).reshape(-1, 1)  # this solution is assumed

# plot the solution
plt.figure(1)
plt.plot(x, u, 'r')
plt.xlabel('x')
plt.ylabel('displacement u')
plt.title('system simulation')

# %% define the dirichlet boundary conditions
u_0 = 1  # displacement at x=0
u_1 = 0  # displacement at x=1

x_B = np.array([0, 1]).reshape(-1, 1)  # boundary points
u_B = np.array([u_0, u_1]).reshape(-1, 1)  # displacements at the boundary

# %% define the collocation points
x_F = x[1:-1, :].reshape(-1, 1)  # take all the elements except the first and the last (boundary points)

# compute the distributed load p for the collocation points, this is a part of the differential equation
p_x_F = (4 * math.pi * math.pi * np.sin(2 * math.pi * x_F)).reshape(-1, 1)

# compute EA
EA = np.ones(x_F.shape[0]).reshape(-1, 1)# %%

# %% define the custom physics loss function
def custom_physics_loss_wrapper(params):
    x_F, p_x_F, EA = params

    # must be defined with these input arguments - template must be followed for this to work
    def custom_physics_loss(X_train, y_train, model):
        """
        Here we will use the physics information (differential equation) to steer the network towards making
        correct predictions for unseen / unlabelled data.

        :return: This function must return a scalar tensor which is the physics loss ! A 0D tensor physical loss value.
        """

        # compute the second derivative using automatic differentiation
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_F)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_F)
                u = model.predict(x_F)  # compute the neural network output for collocation points X_F

            du_dx = tape.gradient(u, x_F)

        d2u_dx2 = tape2.gradient(du_dx, x_F)

        # compute the left side of the differential equation
        EA_d2u_dx2 = tf.multiply(EA, d2u_dx2)
        F = tf.add(EA_d2u_dx2, p_x_F)
        # mse_F = tf.reshape(tf.math.reduce_mean(tf.math.square(F)), [-1])
        mse_F = tf.math.reduce_mean(tf.math.square(F))

        return mse_F

    return custom_physics_loss




# %% define the physics guided neural network
# xavier = tf.keras.initializers.GlorotUniform()  # kernel initializer
xavier = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

# model architecture replicated from the text book example
# using an MLP
model = PGNN(
    tf.keras.layers.Dense(40, kernel_initializer=xavier, activation=tf.nn.sigmoid,
                          input_shape=[x_B.shape[1]]),  # TODO: This shape should be x_B.shape[1]
    # tf.keras.layers.Dense(x_F.shape[0], kernel_initializer=xavier, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, kernel_initializer=xavier),
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=1.0),
    physics_loss_function=custom_physics_loss_wrapper(
        [tf.constant(x_F, dtype=tf.float32), tf.constant(p_x_F, dtype=tf.float32), tf.constant(EA, dtype=tf.float32)]),
    # physics_loss=None,
    lambda_default=1.0,  # bc weight is ten times that of physics loss function
    lambda_physics=0.1
)

# %% train the pgnn
# train the pgnn model in open loop
model.train_model(tf.constant(x_B, dtype=tf.float32), tf.constant(u_B, dtype=tf.float32), num_epochs=5000)
# model.plot_training_loss()  # plot the evolution of the training loss with the epochs
# model.plot_physics_loss()
# model.plot_default_loss()

# %% test the pgnn
'''
Open loop prediction for the training data !
'''

# plot the model prediction for the training dataset
u_pred = model.predict(tf.constant(x, dtype=tf.float32)).numpy()

# plot comparison
plt.figure(2)
plt.plot(x, u, 'r')
plt.plot(x, u_pred, 'b--')
plt.title('open loop test')
plt.xlabel('x')
plt.ylabel('displacement')
plt.legend(['u', 'u_pred'])
plt.grid()
plt.show()
