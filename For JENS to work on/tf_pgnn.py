# -*- coding: utf-8 -*-
# @Author  : Kashan Zafar
# @Email : kashan.syed@tu-dortmund.de
# @Software: PyCharm

"""
This file contains the implementation of the PGNN class. It has been made generic which means that it can be
used to create both mlps and lstms using keras layers. For more information about how to define Keras layers
please refer to the following link. (https://keras.io/api/layers/)
"""

# TODO: Add the ability to save the model to the drive and then load it somehow.
# TODO: Implement stopping criteria when loss decrease is not significant.
# TODO: Switch over from python lists to numpy arrays for internal vectors.


import time
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to suppress tensor flow logs
tf.config.run_functions_eagerly(True)


# List of hyper-parameters
# learning rate
# regularization

# Definition of the PGNN class
class PGNN:
    """
        This class implements a physics guided neural network using keras and tensor flow. Users can use Keras layers but
        this class as the added feature of passing a physics loss function during class instantiation. This custom physics
        loss function will be called in each training epoch and the physics loss along with the loss on the training
        data will be used to train the model (adjust the weights). In the physics loss function the physics information
        such as mathematical relationships can be used to guide the network training all to achieve the goal of obtaining
        a network that can predict physically consistent results on unseen data.

        Parameters
        ----------

        :param layers: The Keras Layers of the neural network.

        This determines the architecture. These layers are kept in the form of a list as an internal variable of the
        class called network_layers. Please refer to one of the examples to learn more about layer definitions. These
        are done by defining Keras layers in the order that they are intended to be arranged in the network.

        :param optimizer: This is the keras optimization algorithm used to train the model.

        Please refer to the following link to learn more about keras optimizers. (https://keras.io/api/optimizers/)
        Supported optimizers are SGD, RMSprop Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl.

        :param default_loss_function: This is the default loss function that will be used during training.

        It will be computed at all times independent of whether the custom physics loss is used or not. This is primarily the
        quantity that the network will try to minimize for each set of training data (X_train) and targets (y_train).
        Learn more about Keras losses here. (https://keras.io/api/losses/). The PGNN class supports three losses which
        are defined using the following strings, 'mse' = mean squared error, 'mae' = mean absolute error,
        'cce' = categorical crossentropy

        :param physics_loss_function: This is the custom physics loss function.

        This is used to enforce physics based constraints in the loss function. This function is called to get the physics loss in each epoch during
        model training. The physics loss is then added to the default loss calculated using the training data.
        The physics_loss is called in this class using three arguments (X_train,y_train,model). Note that a specific
        template has to be followed when defining the custom physics loss.

        :param lambda_default: This is the weightage given to the default loss function.

        After calculation of the default loss it is multiplied by this regularization term. Can be used to alter the importance of the default loss in
        comparison to the custom physics loss.

        :param lambda_physics: This is the weightage given to the custom physics loss function.

        After calculation of the custom physics loss it is multiplied by this regularization term. Can be used to alter the importance of the
        custom physics loss in comparison to the default loss.

        :param early_stop_limit: This parameter defines the stopping limit for the training process for the network.

        When the loss drops below this value the training will be prematurely ended before the specified number of epochs
        have passed.

    """

    def __init__(self, *layers, optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), default_loss_function='mse',
                 physics_loss_function=None,
                 lambda_default=1, lambda_physics=1, early_stop_limit=2e-6, apply_filter = False):

        self.lambda_physics = lambda_physics  # physics loss regularization term
        self.lambda_default = lambda_default  # default loss regularization term

        # user supplied network architecture with keras layers

        self.network_layers = []  # this list will store all the network layers (keras layers)
        for layer in layers:
            self.network_layers.append(layer)  # store the layers in the class variable (stored in a list)

        # optimizer and loss functions

        self.optimizer = optimizer  # set the optimizer

        # set the default loss functions

        if default_loss_function == 'mse':
            self.default_loss_function = tf.keras.losses.MeanSquaredError()

        elif default_loss_function == 'mae':
            self.default_loss_function = tf.keras.losses.MeanAbsoluteError()

        elif default_loss_function == 'cce':
            self.default_loss_function = tf.keras.losses.CategoricalCrossentropy()

        else:
            self.default_loss_function = tf.keras.losses.mean_squared_error  # by default the mean squared error !

        # set the custom physics loss function if one was supplied, else this is set to none
        self.physics_loss_function = physics_loss_function

        # should the PGNN loss be filtered or not?
        self.apply_PGNN_filter = apply_filter

        # stop flag if the loss becomes too small
        self.early_stop_limit = early_stop_limit
        self.premature_stop_flag = False  # this will be set to true when the loss becomes lower than the limit.

        # lists to store the epochs and the losses calculated during the training !
        # these are set to None by default when the model is not trained. the arrays are initialised when training
        # starts

        # the training epoch numbers will be stored in this vector. this will later be useful in plotting the evolution
        # of the losses below
        self.epoch_array = np.array([], dtype=int)
        # this stores the total / overall loss of the neural network during training
        self.training_loss_array = np.array([], dtype='f')
        # this stores the default loss which is the root mean squared error
        self.default_loss_array = np.array([], dtype='f')
        # this stores the physics loss during training
        self.physics_loss_array = np.array([], dtype='f')

    # setters
    def set_physics_loss_function(self, physics_loss_function):
        """
        Set the custom physics loss function that will be used for loss calculation of this network along with the default
        loss during the training phase.

        Parameters
        ----------

        :param physics_loss_function: A python function object for the custom loss.

        This must follow the template of the custom loss function inorder for this to work without any errors.

        """
        self.physics_loss_function = physics_loss_function

    def init_arrays(self):
        """
        Class function to initialize internal storage arrays of the class.
        """
        # the training epoch numbers will be stored in this vector. this will later be useful in plotting the evolution
        # of the losses below
        self.epoch_array = np.array([], dtype=int)
        # this stores the total / overall loss of the neural network during training
        self.training_loss_array = np.array([], dtype='f')
        # this stores the default loss which is the root mean squared error
        self.default_loss_array = np.array([], dtype='f')
        # this stores the physics loss during training
        self.physics_loss_array = np.array([], dtype='f')

    # Running the model, compute the output, can also be used for prediction after training
    def predict(self, X):
        """
        Runs the model for a given input by passing the input manually through layers and returns the output
        of the final layer. Makes a prediction on input features. This will compute a forward pass through the network

        :param X: Given input (as a tensorflow tensor). A network forward pass will be computer for this with this as
        the input. Please ensure that the given tensor is compatible with the input dimensions of the network.

        Returns
        -------

        X : tf.tensor
            Returns the output of all the layers. In short this is the output after one forward pass through the network.

        """
        # feed X though the network and obtain the network output
        for layer in self.network_layers:

            if layer.__class__.__name__ == 'LSTM':  # this needs to be done since lstms accept 3D inputs
                # TODO: check dimension expansion axis
                X = tf.expand_dims(X, 2)  # if using multiple lstm units
                # X = tf.expand_dims(X, 1)  # if using a single lstm unit

            X = layer(X)

        return X

    # Custom loss function
    def get_loss(self, X_train, y_train):
        """
        Computes the loss and returns it as a TF EagerTensor value. If incase you make changes please make sure you use
        only tensor flow operations. This function will compute both the default loss and if applicable the custom physics
        loss. It will then multiply them with their respective weights (regularization terms) and return the sum in the
        return value called training loss.

        Parameters
        ----------
        :param X_train: The tensor containing the training examples. Examples are in rows and the features are in columns.

        :param y_train: The tensor containing the targets that the network is expected to learn to predict.

        Returns
        -------
        training_loss : tf.tensor
            Sum of the NN loss function comparing the y_predicted against
            y_true and the physical loss function (self._p_fun) with
            respective weights applied.
        default_loss : tf.tensor
            Standard NN training loss comparing y_train to y_predicted where y_predicted is the prediction of the
            network for the training data. The type of loss is defined during class instantiation.
        physics_loss : tf.tensor
            Physics loss from physics_loss_function after regularization.
        """

        # create empty placeholder variables
        training_loss = tf.constant(0.0, dtype=tf.float32)
        default_loss = tf.constant(0.0, dtype=tf.float32)
        physics_loss = tf.constant(0.0, dtype=tf.float32)

        if self.lambda_default != 0.0:
            nn_op = self.predict(X_train)  # compute the neural network output for the given X
            # compute the default loss which is the root means square error
            default_loss = self.default_loss_function(nn_op, y_train)
            # regularize the default loss !
            default_loss = tf.math.scalar_mul(self.lambda_default, default_loss)
            training_loss += default_loss

        # if physics loss has been set for the class then add custom physics loss as well to the training loss
        if self.physics_loss_function is not None:
            physics_loss = self.physics_loss_function(X_train, y_train, self)  # call the custom physics loss function

            # if not tf.is_tensor(physics_loss):
            #     raise Exception('\nPhysics loss must be returned as a tensor flow scalar tensor ! '
            #                     'Please check your custom '
            #                     'physics loss implementation !')

            #physics_loss = tf.math.scalar_mul(self.lambda_physics, physics_loss)
            #training_loss += #physics_loss

        if tf.math.is_nan(training_loss):
            msg = 'PGNN calculated a NaN loss value!'
            raise ArithmeticError(msg)

        return training_loss, default_loss, physics_loss

    # get gradients
    def get_gradient(self, X_train, y_train):
        """
        Till now it seemed pretty easy to implement loss since we were dealing directly with models, but now we need to
        perform learning by auto-differentiation. This is implemented in TF 2.0 using tf.GradientTape(). The function
        get_grad() computes the gradient wrt to the variables of the layers. It is important to note that all the
        variables and arguments are tensorflow tensors. For more information about auto differentiation please refer
        to the following tensorflow guide. (https://www.tensorflow.org/guide/autodiff?hl=en)

        Parameters
        ----------

        :param X_train: The tensor containing the training examples. Examples are in rows and the features are in columns.

        :param y_train: The tensor containing the targets that the network is expected to learn to predict.

        Returns
        -------

        gradient : tf.tensor
            The gradient of the loss wrt to the network variables (weights and biases).
        layer_variables_list : python list
            The list of tf.Variable objects. These are the trainable variables of the network
        training_loss : tf.tensor
            The sum of the regularized default loss and the custom physics loss.
        default_loss : tf.tensor
            Standard NN training loss comparing y_train to y_predicted where y_predicted is the prediction of the
            network for the training data. The type of loss is defined during class instantiation.
        physics_loss : tf.tensor
            Physics loss from physics_loss_function after regularization.

        """


        with tf.GradientTape(persistent=True) as tape:
            # calculate the loss for the current epoch
            training_loss, default_loss, physics_loss = self.get_loss(X_train, y_train)

            layer_variables_list = []  # hold the layer variables with respect to which differentiation will be done
            for layer in self.network_layers:
                # tape.watch(layer.variables)
                for variable in layer.variables:
                    tape.watch(variable)  # first you need to watch all the variables in all of the layers
                    layer_variables_list.append(variable)  # ensure that all the layer variables are appended !



            # perform auto differentiation
            #gradient = tape.gradient(training_loss, layer_variables_list)
            defaultGradient  = tape.gradient(default_loss, layer_variables_list)
            
            physicsGradientList = []
            if self.physics_loss_function is not None:
                for L_i in physics_loss:
                    if (L_i > 0) or not self.apply_PGNN_filter:
                         dL_dw_lists = tape.gradient(L_i, layer_variables_list)
                         physicsGradientList.append(dL_dw_lists)
            # physicsGraident  = tape.graident(physics_loss, layer_variables_list)
            # print(physicsGradientList)
            gradient = defaultGradient
            if True: #self.physics_loss_function is not None:
                if len(physicsGradientList) > 0:
                    for iList in physicsGradientList:
                        for iL in range(len(iList)):
                            gradient[iL] += iList[iL]



            # # Addition of the two gradient lists
            # addition_of_physics_losses_l1_l2 = []
            # addition_of_physics_default_losses = []
            #
            # # for i in len(physicsGradientList):
            # # for i in range(0, len(physicsGradientList)):
            # # x = x.append(physicsGradientList[i])
            #
            # if len(physicsGradientList) ==0:
            #
            #     addition_of_physics_default_losses = defaultGradient
            #
            #     # for i in range(0, 10):
            #     #     # adding L_def + L_phy1 + L_phy2
            #     #     physicsGradientList_plus_defaultGradient = defaultGradient[i]
            #     #     addition_of_physics_default_losses.append(physicsGradientList_plus_defaultGradient)
            #     print(addition_of_physics_default_losses)
            #
            # elif len(physicsGradientList) ==1:
            #     # for i in range(0, 10):
            #     #     physicsGradientList_plus_physics_losses = physicsGradientList[0][i]
            #     #     addition_of_physics_losses_l1_l2.append(physicsGradientList_plus_physics_losses)
            #     # addition_of_physics_losses_l1_l2 = physicsGradientList[0]
            #
            #     for i in range(0, 10):
            #         # adding L_def + L_phy1 + L_phy2
            #         physicsGradientList_plus_defaultGradient =physicsGradientList[0][i] + defaultGradient[i]
            #         addition_of_physics_default_losses.append(physicsGradientList_plus_defaultGradient)
            #     print(addition_of_physics_default_losses)
            #
            #
            # else:
            #     # Adding all of the losses from the physics constraints that were differentiated the L with respect to p:
            #     addition_of_physics_losses_l1_l2 = [sum(i) for i in zip(*physicsGradientList)]
            #
            #     for i in range(0, 10):
            #         # adding L_def + L_phy1 + L_phy2
            #         physicsGradientList_plus_defaultGradient =addition_of_physics_losses_l1_l2[i] + defaultGradient[i]
            #         addition_of_physics_default_losses.append(physicsGradientList_plus_defaultGradient)
            #     print(addition_of_physics_default_losses)
            #
            #
            #
            # # gradient = defaultGradient
            # gradient = addition_of_physics_default_losses
        return gradient, layer_variables_list, training_loss, default_loss, physics_loss

    # perform gradient descent
    def network_learn(self, X_train, y_train, epoch_num):
        """
        This function performs a single step of gradient descent. It first computes the gradients and then uses the
        optimizer to adjust the weights of the network thereby completing a training step.

        Parameters
        ----------

        :param X_train: The tensor containing the training examples. Examples are in rows and the features are in columns.

        :param y_train: The tensor containing the targets that the network is expected to learn to predict.

        :param epoch_num: The current epoch integer. This will be printed out in the function.

        Returns
        -------

        training_loss : tf.tensor
            The sum of the regularized default loss and the custom physics loss.
        default_loss : tf.tensor
            Standard NN training loss comparing y_train to y_predicted where y_predicted is the prediction of the
            network for the training data. The type of loss is defined during class instantiation.
        physics_loss : tf.tensor
            Physics loss from physics_loss_function after regularization.

        """
        # We apply the gradient descent step in this function using the gradients obtained from
        # the get_grad() function.

        # get the gradient for this epoch
        gradient, layer_variables_list, training_loss, default_loss, physics_loss = self.get_gradient(X_train,
                                                                                                      y_train)

        # apply the gradients to the layer variables / neuron weights
        self.optimizer.apply_gradients(zip(gradient, layer_variables_list))

        print('\nEpoch: ' + str(epoch_num) + ' - Default Loss: ' + str(training_loss) + ' - Physics Loss: ' + str(
            physics_loss))

        return training_loss, default_loss, physics_loss  # return the loss so that it can be plotted.

    def plot_loss(self, loss_array, ylabel='', color='m', title=''):
        """
        This function plots the evolution of the specified loss function against the training epochs.

        Parameters
        ----------

        :param loss_array: The loss array that the user wishes to plot.
        :param ylabel: The label for the y-axis.
        :param color: The color of the plot.
        :param title: The title of the plot.

        """
        # plot the training loss of the training session
        fig = plt.figure(random.randint(1, 1000))
        plt.plot(self.epoch_array, loss_array, color)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_training_loss(self):
        """
        This function will plot the evolution of the combined loss function during the training using matplotlib. This
        can be useful in visualizing the training loss during the training process.
        """

        if self.epoch_array.size != 0 and self.training_loss_array.size != 0:
            self.plot_loss(self.training_loss_array, ylabel='Training Loss', color='m',
                           title='Training Loss (Default + Physics) vs Epochs')
        else:
            print('\nTraining Loss array is empty ! Please train the model first !')

    def plot_physics_loss(self):
        """
        This function will plot the evolution of the physics loss function during the couse of the training if in case
        a custom physics loss function is specified.
        """

        if self.epoch_array.size != 0 and self.physics_loss_array.size != 0:
            self.plot_loss(self.physics_loss_array, ylabel='Physics Loss', color='m',
                           title='Physics Loss vs Epochs')
        else:
            print('\nPhysics Loss array is empty ! Please train the model first !')

    def plot_default_loss(self):
        """
        This function will plot the evolution of the default loss function over the training epochs
        """

        if self.epoch_array.size != 0 and self.default_loss_array.size != 0:
            # TODO: Add the feature to mention the type of loss in the plot here (save string)
            self.plot_loss(self.default_loss_array, ylabel='Default Loss', color='m',
                           title='Default Loss vs Epochs')
        else:
            print('\nDefault Loss array is empty ! Please train the model first !')

    def train_model(self, X_train, y_train, num_epochs=100):
        """
        This important PGNN class function should be called to train the neural network to predict the targets for the
        training examples. It is like the fit() method for a Keras model. Calling this function will commence the training
        process for the physics guided neural network.

        Parameters
        ----------

        :param X_train: The tensor containing the training examples. Examples are in rows and the features are in columns.

        :param y_train: The tensor containing the targets that the network is expected to learn to predict.

        :param num_epochs: The total number of epochs for the training.

        The gradient descent and update will be performed this many number of times. Defaults to 100.

        """
        self.init_arrays()  # reset internal storage arrays

        # In the code section below we convert check that the variables X_train and y_train are tensor flow tensors
        # since this class best works with tensors.

        if not tf.is_tensor(X_train):
            raise TypeError('X_train must be a tf.Tensor')

        if not tf.is_tensor(y_train):
            raise TypeError('y_train must be a tf.Tensor')

        start = time.process_time()
        # train the model for the number of epochs
        for i in range(num_epochs):
            training_loss, default_loss, physics_loss_list = self.network_learn(X_train, y_train, i)
            training_loss = training_loss.numpy()  # extract the loss value from the tensor
            self.training_loss_array = np.append(self.training_loss_array, training_loss)
            physics_loss = 0.0
            if self.physics_loss_function is not None:
                for pL_i in physics_loss_list:
                    physics_loss += pL_i.numpy()  # extract the loss value from the tensor
            self.physics_loss_array = np.append(self.physics_loss_array, physics_loss)

            default_loss = default_loss.numpy()  # extract the loss value from the tensor
            self.default_loss_array = np.append(self.default_loss_array, default_loss)

            self.epoch_array = np.append(self.epoch_array, i)

            if training_loss < self.early_stop_limit:
                print('\nLoss value too small. Stopping training early prematurely !')
                break
        tic_toc = str(time.process_time() - start)
        print('\nPGNN Model trained successfully !' + ' Epochs Completed: ' + str(self.epoch_array[-1]))
        print('\nTime taken to train: ' + tic_toc)

        return default_loss, tic_toc
