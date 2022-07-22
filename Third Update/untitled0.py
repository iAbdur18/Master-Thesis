# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:07:46 2022

@author: Abdur Rehman
"""

import matplotlib

matplotlib.use('Qt5Agg')
import tensorflow as tf

tf.config.run_functions_eagerly(True)

import math
import numpy as np
import matplotlib.pyplot as plt

# custom imports
from tf_pgnn import PGNN

# %% create the training and test data
# x points used for testing
delta_x = 0.2
points_range = 3
x_test = np.arange(0, points_range + delta_x, delta_x).reshape(-1, 1)  # randomly generate numbers between 0 and points range

# the corresponding function values for the X_test points
y_test = np.exp(x_test)

# first half of the points will be used for training. The other half will be collocation points for the custom loss function
x_B, x_F = np.array_split(x_test, 2)  # boundary points and collocation points
y_B, y_F = np.array_split(y_test, 2)

# %% define a neural network (without the custom physics loss function)
xavier = tf.keras.initializers.GlorotUniform()  # kernel initializer
# using an MLP
model = PGNN(
    tf.keras.layers.Dense(10, kernel_initializer=xavier, activation=tf.nn.sigmoid,
                          input_shape=[x_B.shape[1]]),
    # tf.keras.layers.Dense(x_F.shape[0], kernel_initializer=xavier, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, kernel_initializer=xavier),
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.7),
    physics_loss_function=None,
    lambda_default=1.0,
    lambda_physics=0.0
)

# %% train the pgnn
# this nn will be trained using only the first half of the data
model.train_model(tf.constant(x_B, dtype=tf.float32), tf.constant(y_B, dtype=tf.float32), num_epochs=1000)
# model.plot_training_loss()  # plot the evolution of the training loss with the epochs
# model.plot_physics_loss()

# %% test the pgnn
'''
Open loop prediction for the training data !
'''

# plot the model prediction for the complete data (all the x values)
y_pred = model.predict(tf.constant(x_test, dtype=tf.float32)).numpy()

# plot comparison
plt.figure(2)
plt.plot(x_test, y_test, 'r')
plt.plot(x_test, y_pred, 'b--')
plt.title('open loop test')
plt.xlabel('x')
plt.ylabel('y=exp(x)')
plt.legend(['y=exp(x)', 'y_pred nn'])
plt.grid()
plt.show()

# %% define a custom physics loss function that can be used with a pgnn
def custom_physics_loss_wrapper(params):
    x_F = params  # the collocation points are passed into the cost function as parameters

    # must be defined with these input arguments - template must be followed for this to work
    def custom_physics_loss(X_train, y_train, model):
        """
        Here we will use the physics information (differential equation) to steer the network towards making
        correct predictions for unseen / unlabelled data.
        """

        # compute the second derivative using automatic differentiation
        with tf.GradientTape() as tape2:
            tape2.watch(x_F)

            with tf.GradientTape() as tape:
                tape.watch(x_F)
                y = model.predict(x_F)  # compute the neural network output for collocation points x_F

            dy_dx = tape.gradient(y, x_F)  # compute the first derivative

        d2y_dx2 = tape2.gradient(dy_dx, x_F)  # compute the second derivative

        # approach 1
        # ensure monotonicity on both the first and the second derivative
        # relu the negative of the derivative of the network
        mse_1 = tf.math.reduce_sum(tf.nn.relu(tf.math.scalar_mul(-1, dy_dx)))
        mse_2 = tf.math.reduce_sum(tf.nn.relu(tf.math.scalar_mul(-1, d2y_dx2)))

        # mse_F = mse_1 + mse_2 # add the two mses and convert to a scaler tensor

        # approach 2
        # ensure that the derivative and the predictions match
        # dy_dy must be same as y
        # mse_1 = tf.reshape(tf.math.reduce_sum(tf.math.abs(tf.math.subtract(dy_dx, y))), [-1])
        # mse_2 = tf.reshape(tf.math.reduce_sum(tf.math.abs(tf.math.subtract(d2y_dx2, y))), [-1])

        return tf.math.add(mse_1, mse_2)

    return custom_physics_loss

# %% define the pgnn with the custom loss function
xavier = tf.keras.initializers.GlorotUniform()  # kernel initializer

# using an MLP
model = PGNN(
    tf.keras.layers.Dense(10, kernel_initializer=xavier, activation=tf.nn.relu,
                          input_shape=[x_B.shape[1]]),
    # tf.keras.layers.Dense(x_F.shape[0], kernel_initializer=xavier, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, kernel_initializer=xavier),
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=1.0),
    physics_loss_function=custom_physics_loss_wrapper(tf.constant(x_F, dtype=tf.float32)),
    lambda_default=1.0,
    lambda_physics=5.0
)

# %% train the pgnn
# this nn will be trained using only the first half of the data
model.train_model(tf.constant(x_B, dtype=tf.float32), tf.constant(y_B, dtype=tf.float32), num_epochs=10000)
# model.plot_training_loss()  # plot the evolution of the training loss with the epochs
# model.plot_physics_loss()

# %% test the pgnn
'''
Open loop prediction for the training data !
'''

# plot the model prediction for the complete data (all the x values)
y_pred_pgnn = model.predict(tf.constant(x_test, dtype=tf.float32)).numpy()

# plot comparison
plt.figure(3)
plt.plot(x_test, y_test, 'r')
plt.plot(x_test, y_pred, 'b--')
plt.plot(x_test, y_pred_pgnn, 'g--')
plt.title('open loop test')
plt.xlabel('x')
plt.ylabel('y=exp(x)')
plt.legend(['y=exp(x)', 'y_pred nn', 'y_pred_pgnn'])
plt.grid()
plt.show()