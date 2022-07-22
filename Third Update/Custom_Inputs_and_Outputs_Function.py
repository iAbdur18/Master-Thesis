# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 17:14:35 2022

@author: Abdur Rehman
"""
from NARX_Concatenation_Function_Updated import concateDataForNARX

# %%
print('Custom Inputs and Outputs Function Being Called')
# Need to define a function that takes inputs, outputs and delays and returns
# X_train, X_test, y_test, y_train, input_shape, output_shape


def custom_inputs_outputs(simulated_input_3d, concatenate_array_output, input_output_delays):
    # Normalization of the DATA Received
    print('Normalization of the DATA')
    
    print('Normalization of the Inputs')
    df_max_scaled_input = simulated_input_3d.copy()

    # apply normalization techniques
    for column in df_max_scaled_input.columns:
        df_max_scaled_input[column] = df_max_scaled_input[column] / \
            df_max_scaled_input[column].abs().max()

    print('Normalization of the Outputs')
    df_max_scaled_output = concatenate_array_output.copy()

    # apply normalization techniques
    for column in df_max_scaled_output.columns:
        df_max_scaled_output[column] = df_max_scaled_output[column] / \
            df_max_scaled_output[column].abs().max()

    # Delays of the Inputs and Outputs
    nU_delay = int(input_output_delays)
    nY_delay = int(input_output_delays)

    [inputs, targets] = concateDataForNARX(
        df_max_scaled_input, df_max_scaled_output, nU_delay, nY_delay)

    # MACHINE LEARNING
    inputs_for_machine_learning = inputs
    outputs_for_machine_learning = targets

    # Setting of the Training and Testing Sets
    X_train = inputs_for_machine_learning.iloc[:3000]
    X_test = inputs_for_machine_learning.iloc[3000:]

    y_train = targets.iloc[:3000]
    y_test = targets.iloc[3000:]

    # Defining the Input and Output layers based on the NARX Input and Output
    input_layers = int((nU_delay*2)+(nY_delay * 4))
    output_layers = int(4)
    return X_train, X_test, y_train, y_test, input_layers, output_layers
