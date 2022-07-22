# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:49:13 2022

@author: Abdur Rehman
"""
import numpy as np
from Continuous_Stirred_Tank_Reactor import simulator
# %%

bbb = np.array([100, -8500])

bbb.reshape(bbb.shape[0],-1)

bbb = bbb.reshape(bbb.shape[0],-1)

ccc = simulator.make_step(np.array(bbb))
print(ccc)