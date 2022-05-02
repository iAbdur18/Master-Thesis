# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:21:39 2022

@author: Abdur Rehman
"""


# DATA EXTRACTION

# Function defined to extract the data from the simulator defined above,
# It taks the iteration steps, iteartion length and the range of the input
# It is defined for the random input signals in the simulator

def random_number_generator(nstep_function, sym_len_function, a_range_function):
  import numpy as np
  nstep = nstep_function
  sym_len = sym_len_function
  a_range = a_range_function

  max_num = max(a_range)
  min_num = min(a_range)
  range_series = [max_num - min_num]
  range_series = range_series[0]

  # range for amplitude
  a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0]
  a = np.round(a)
  a = a.astype(int)
  sig = np.zeros(nstep*sym_len)
  i = 0
  for i in range(i, np.size(a)):
      temp = int(i*sym_len)
      sig[temp:temp+sym_len] = a[i]
  return sig


# =============================================================================
