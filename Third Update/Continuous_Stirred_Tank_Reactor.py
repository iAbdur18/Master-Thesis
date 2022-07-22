# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 09:50:05 2022

@author: Abdur Rehman
"""

# CONTINUOUS STIRRED TANK REACTOR (CSTR)

# MODEL CREATION

# =============================================================================
# %%
# STEP 1: Importing Basic Modules And Do-Mpc
import pandas as pd
import matplotlib.pyplot as plt
import do_mpc
from casadi import *
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from Random_Number_Generator_Updated import random_number_generator
from NARX_Concatenation_Function_Updated import concateDataForNARX
print('Importing Basic Modules And Do-Mpc')


# Import do_mpc package:


# STEP 2: Model Initialization In do_mpc

model_type = 'continuous'  # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)


# Step 3: States and Control Inputs

# States:
# The four states are concentration of reactant A (CA), the concentration of
# reactant B (CB), the temperature inside the reactor (TR) and the temperature
# of the cooling jacket (TK)

# States struct (optimization variables):
C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1, 1))
C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1, 1))
T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1, 1))
T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1, 1))

# Inputs
# The control inputs are the feed F and the heat flow Q_DOT
# Input struct (optimization variables):
F = model.set_variable(var_type='_u', var_name='F')
Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

# Step 3: ODE and Parametersc

# The parameters α and β are uncertain while the rest of the parameters is
# considered certain

# Certain parameters
K0_ab = 1.287e12  # K0 [h^-1]
K0_bc = 1.287e12  # K0 [h^-1]
K0_ad = 9.043e9  # K0 [l/mol.h]
R_gas = 8.3144621e-3  # Universal gas constant
E_A_ab = 9758.3*1.00  # * R_gas# [kj/mol]
E_A_bc = 9758.3*1.00  # * R_gas# [kj/mol]
E_A_ad = 8560.0*1.0  # * R_gas# [kj/mol]
H_R_ab = 4.2  # [kj/mol A]
H_R_bc = -11.0  # [kj/mol B] Exothermic
H_R_ad = -41.85  # [kj/mol A] Exothermic
Rou = 0.9342  # Density [kg/l]
Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
A_R = 0.215  # Area of reactor wall [m^2]
V_R = 10.01  # 0.01 # Volume of reactor [l]
m_k = 5.0  # Coolant mass[kg]
T_in = 130.0  # Temp of inflow [Celsius]
K_w = 4032.0  # [kj/h.m^2.K]
# Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
C_A0 = (5.7+4.5)/2.0*1.0

# Uncertain parameters:
alpha = model.set_variable(var_type='_p', var_name='alpha')
beta = model.set_variable(var_type='_p', var_name='beta')

# In the next step, we formulate the ki-s
# Auxiliary terms
K_1 = beta * K0_ab * exp((-E_A_ab)/((T_R+273.15)))
K_2 = K0_bc * exp((-E_A_bc)/((T_R+273.15)))
K_3 = K0_ad * exp((-alpha*E_A_ad)/((T_R+273.15)))

# define an artificial variable of interest
T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

# Can define the ODEs
model.set_rhs('C_a', F*(C_A0 - C_a) - K_1*C_a - K_3*(C_a**2))
model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2) *
              H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) + (((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))

# Step 4: Finally, the model setup is completed
# Build the model
model.setup()

# %%
# =============================================================================

# CONTROLLER DESIGN
print('Configure the controller')
# Step 1: Configure the controller
# member of the mpc class is generated with the prediction model defined above

mpc = do_mpc.controller.MPC(model)

# Step 2: Define parameters of the discretization scheme orthogonal collocation
setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 0.005,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

# Step 3: Scaling of the states and inputs

mpc.scaling['_x', 'T_R'] = 100
mpc.scaling['_x', 'T_K'] = 100
mpc.scaling['_u', 'Q_dot'] = 2000
mpc.scaling['_u', 'F'] = 100

# Step 4: Objective
# The goal of the CSTR is to obtain a mixture with a concentration of
# CB,ref=0.6 mol/l. Additionally, we add a penalty on input changes for both
# control inputs, to obtain a smooth control performance.
_x = model.x
mterm = (_x['C_b'] - 0.6)**2  # terminal cost
lterm = (_x['C_b'] - 0.6)**2  # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(F=0.1, Q_dot=1e-3)  # input penalty

# Step 4: Constraints

# lower bounds of the states
mpc.bounds['lower', '_x', 'C_a'] = 0.1
mpc.bounds['lower', '_x', 'C_b'] = 0.1
mpc.bounds['lower', '_x', 'T_R'] = 50
mpc.bounds['lower', '_x', 'T_K'] = 50

# upper bounds of the states
mpc.bounds['upper', '_x', 'C_a'] = 2
mpc.bounds['upper', '_x', 'C_b'] = 2
mpc.bounds['upper', '_x', 'T_K'] = 140

# lower bounds of the inputs
mpc.bounds['lower', '_u', 'F'] = 5
mpc.bounds['lower', '_u', 'Q_dot'] = -8500

# upper bounds of the inputs
mpc.bounds['upper', '_u', 'F'] = 100
mpc.bounds['upper', '_u', 'Q_dot'] = 0.0


# Step 5: Soft Constraints
mpc.set_nl_cons('T_R', _x['T_R'], ub=140,
                soft_constraint=True, penalty_term_cons=1e2)

# Step 6: Uncertain Values
alpha_var = np.array([1., 1.05, 0.95])
beta_var = np.array([1., 1.1, 0.9])

mpc.set_uncertainty_values(alpha=alpha_var, beta=beta_var)

# Step 7: Setup the Controller
mpc.setup()
# %%
# =============================================================================

# Estimatior
estimator = do_mpc.estimator.StateFeedback(model)

# =============================================================================

# %%

# Simulator
# Step 1: create an instance of the do-mpc simulator which is based on
# the same model
simulator = do_mpc.simulator.Simulator(model)

# Step 2: Define necessary parameters
params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.005
}


simulator.set_param(**params_simulator)


# Step 3: Define uncertain paramters
p_num = simulator.get_p_template()
tvp_num = simulator.get_tvp_template()

# function for time-varying parameters


def tvp_fun(t_now):
    return tvp_num


# uncertain parameters
p_num['alpha'] = 1
p_num['beta'] = 1


def p_fun(t_now):
    return p_num


simulator.set_tvp_fun(tvp_fun)
simulator.set_p_fun(p_fun)

# Step 4: Setup
simulator.setup()

# =============================================================================
# %%
# Closed-loop simulation
print('Closed-loop simulation of the Controller')
# Step 1: Set Initial Guess
# Set the initial state of mpc, simulator and estimator:
C_a_0 = 0.8  # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5  # This is the controlled variable [mol/l]
T_R_0 = 134.14  # [C]
T_K_0 = 130.0  # [C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1, 1)

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

# Step 2: Closed Loop for 50 steps
input_values = []
state_values = []
for k in range(50):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)

    x0 = estimator.make_step(y_next)


# =============================================================================
# Animating the results
# configure the do-mpc graphics object, which is initiated with the respective
# data object
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)


# configure Matplotlib
rcParams['axes.grid'] = True
rcParams['font.size'] = 18

# configure which lines to plot on which axis and add labels
fig, ax = plt.subplots(5, sharex=True, figsize=(16, 12))
# Configure plot:
mpc_graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
mpc_graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
mpc_graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
mpc_graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
mpc_graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')


# Update properties for all prediction lines:
for line_i in mpc_graphics.pred_lines.full:
    line_i.set_linewidth(2)
# Highlight nominal case:
for line_i in np.sum(mpc_graphics.pred_lines['_x', :, :, 0]):
    line_i.set_linewidth(5)
for line_i in np.sum(mpc_graphics.pred_lines['_u', :, :, 0]):
    line_i.set_linewidth(5)
for line_i in np.sum(mpc_graphics.pred_lines['_aux', :, :, 0]):
    line_i.set_linewidth(5)

# Add labels
label_lines = mpc_graphics.result_lines['_x',
                                        'C_a']+mpc_graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = mpc_graphics.result_lines['_x',
                                        'T_R']+mpc_graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()


def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines


n_steps = mpc.data['_time'].shape[0]


anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

gif_writer = ImageMagickWriter(fps=5)
anim.save('anim_CSTR.gif', writer=gif_writer)

# =============================================================================
# %%
# DATA EXTRACTION
print('DATA EXTRACTION')
# Function defined to extract the data from the simulator defined above,
# It taks the iteration steps, iteartion length and the range of the input
# It is defined for the random input signals in the simulator

# =============================================================================

# Function for the concatenation of the NARX DATA
# Concatenation of the DATA to NARX


# =============================================================================
# Extracting the DATA from the Simulator

# Define the number of steps for the random data
nstep_function = 40
# define the duration of the random number to be kept constant for
sym_len_function = 90
# Input F range
F_range_function = [5, 100]
# Input Q_dot range
Q_range_function = [-8500, 0]
input_multiplier = nstep_function*sym_len_function
# Total Number of Outputs
output_length = 4
# Total Number of Inputs
input_length = 2

# define a new variable for the concatenation of both inputs in one array
concatenate_array = np.zeros(2*nstep_function*sym_len_function)
# for the for loop's range
length_of_an_array = int(len(concatenate_array)/2)
# output concatenation array
concatenate_array_output = numpy.zeros((length_of_an_array, output_length, 1))


# calling random number generator function to get the values of both the inputs
var_Q_dot = random_number_generator(
    nstep_function, sym_len_function, Q_range_function)
var_F = random_number_generator(
    nstep_function, sym_len_function, F_range_function)


# Define a for loop to concatenate the two inputs in one array
i = 0
for i in range(0, length_of_an_array):
    k = 2*i
    concatenate_array[k] = var_F[i]
    concatenate_array[k+1] = var_Q_dot[i]

# Converting the received array to 3D
simulated_input_3d = np.reshape(
    concatenate_array, (length_of_an_array, input_length, 1))


# Simulating the output array
for i in range(0, length_of_an_array):
    concatenate_array_output[i] = simulator.make_step(
        simulated_input_3d[:][i][:])

# Rounding the simulated output to 3 decimal places
concatenate_array_output = np.round(concatenate_array_output, 3)

# Changing the 3D Array to 2D array of both the Inputs and Outputs

concatenate_array_output = concatenate_array_output.reshape(
    input_multiplier, 4)
simulated_input_3d = simulated_input_3d.reshape(input_multiplier, 2)


# Changing the 2D Numpy Array to Pandas Dataframe
concatenate_array_output = pd.DataFrame(concatenate_array_output)
simulated_input_3d = pd.DataFrame(simulated_input_3d)

#%%
# Normalization of the DATA Received
print('Normalization of the DATA')
df_max_scaled_input = simulated_input_3d.copy()

# apply normalization techniques
for column in df_max_scaled_input.columns:
    df_max_scaled_input[column] = df_max_scaled_input[column] / \
        df_max_scaled_input[column].abs().max()

# view normalized data
display(df_max_scaled_input)
print(df_max_scaled_input)
print(max(df_max_scaled_input))
print(min(df_max_scaled_input))

#%%

df_max_scaled_output = concatenate_array_output.copy()

# apply normalization techniques
for column in df_max_scaled_output.columns:
    df_max_scaled_output[column] = df_max_scaled_output[column] / \
        df_max_scaled_output[column].abs().max()

# view normalized data
display(df_max_scaled_output)
print(df_max_scaled_output)
print(max(df_max_scaled_output))
print(min(df_max_scaled_output))

#%%

nU_delay = int(2)
nY_delay = int(2)


[inputs, targets] = concateDataForNARX(
    df_max_scaled_input, df_max_scaled_output, nU_delay, nY_delay)
#%%
# Storing the Data in a CSV file
print('Storing the Data in a CSV file')
inputs.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\inputs_simulated_data.xlsx', index=False)

targets.columns = ['Concentration of Reactant A', 'Concentration of Reactant B',
                   'Temperature inside the reactor TR', 'Temperature of the Cooling Jacket TK']

targets.to_excel(
    r'C:\Users\Abdur Rehman\Thesis Code\outputs_simulated_data.xlsx', index=False)

# =============================================================================
#%%

# %%
plt.clf()
print('PLOTS')
plt.plot(simulated_input_3d[:])
plt.title('The Control Inputs')
plt.ylabel('Inputs')
plt.xlabel('Iterations')
plt.legend(['Feed', 'Heat Flow'], loc='upper left')
plt.show()
#%%
plt.plot(simulated_input_3d[0][:100])
plt.title('The Control Inputs')
plt.ylabel('Feed')
plt.xlabel('Iterations')
plt.legend(['Feed'], loc='upper left')
plt.show()

#%%

plt.plot(simulated_input_3d[1][:100])
plt.title('The Control Inputs')
plt.ylabel('Feed')
plt.xlabel('Iterations')
plt.legend(['Heat Flow'], loc='upper left')
plt.show()

#%%

plt.plot(concatenate_array_output[:100])
plt.title('The Four States')
plt.ylabel('States')
plt.xlabel('Iterations')
plt.legend(['reactant A ', 'reactant B',
           'temperature TR', 'temperature TK'], loc='right')
plt.show()

#%%
plt.plot(concatenate_array_output[0][:100])
plt.title('The States')
plt.ylabel('States')
plt.xlabel('Iterations')
plt.legend(['concentration of reactant A (CA)'], loc='upper left')
plt.show()
#%%
plt.plot(concatenate_array_output[1][:100])
plt.title('The States')
plt.ylabel('States')
plt.xlabel('Iterations')
plt.legend(['concentration of reactant B (CB)'], loc='upper left')
plt.show()
#%%
plt.plot(concatenate_array_output[2][:100])
plt.title('The States')
plt.ylabel('States')
plt.xlabel('Iterations')
plt.legend(['temperature inside the reactor (TR)'], loc='upper left')
plt.show()
#%%
plt.plot(concatenate_array_output[3][:100])
plt.title('The States')
plt.ylabel('States')
plt.xlabel('Iterations')
plt.legend(['temperature of the cooling jacket (TK)'], loc='upper left')
plt.show()
#%%
# Inputs
print(inputs.shape)
print(inputs)
plt.plot(inputs[:50])
plt.title('NARX Inputs')
plt.ylabel('NARX Inputs')
plt.xlabel('Iterations')
plt.show()
#%%
# Targets
print(targets.shape)
print(targets)
plt.plot(targets[:50])
plt.title('NARX Targets')
plt.ylabel('NARX Targets')
plt.xlabel('Iterations')
plt.show()

#%%
