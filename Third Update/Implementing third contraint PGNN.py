# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:06:45 2022

@author: Abdur Rehman
"""

# %%


# Implementing third contraint
plt.rcParams["figure.figsize"] = (28, 15)
closed_loop_NParray = np.array(closed_loop_NParray)
time_x_axis_experiment = np.linspace(0, 596, 596)
q_dot_y_axis = closed_loop_NParray[:, 23]
jacket_temperature_y_axis = closed_loop_NParray[:, 3]


#plt.plot(time_x_axis_experiment, q_dot_y_axis, color='green', linestyle='dashed', label=["Q_Dot"])
plt.plot(time_x_axis_experiment, jacket_temperature_y_axis,
         color='blue', linestyle='dashed', label=["Jacket Temperature: TK"])
plt.plot(time_x_axis_experiment, y_test_HyperParametersModel[3], color='red', linestyle='dashed', label=[
         "Jacket Temperature: TK"])
plt.plot(time_x_axis_experiment, y_pred_test_dataframe[3], color='green', linestyle='dashed', label=[
         "Jacket Temperature: TK"])
plt.legend(loc='upper right')
# %%
# Backpropagation and forward propagation
step_difference = int(80)
rows_for_the_loop = int(np.array(q_dot_y_axis.shape))

jacket_temperature_difference = [jacket_temperature_y_axis[(
    i+(1+step_difference))]-jacket_temperature_y_axis[(i+step_difference)] for i in range(rows_for_the_loop-step_difference-1)]
jacket_temperature_difference_numpy = np.array(jacket_temperature_difference)

q_dot_difference = [q_dot_y_axis[(i+step_difference)]-q_dot_y_axis[i]
                    for i in range(rows_for_the_loop-step_difference-1)]
q_dot_difference_numpy = np.array(q_dot_difference)

# %%
# array multiplication
product_q_dot_and_jacket_temperature = np.multiply(
    q_dot_difference_numpy, jacket_temperature_difference_numpy)

# %%
loss_experiment = tf.nn.relu(-product_q_dot_and_jacket_temperature)
# %%
loss_experiment_2 = tf.math.reduce_sum(
    tf.nn.relu(-product_q_dot_and_jacket_temperature))

# %%
