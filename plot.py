#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 04:03:13 2018

@author: katerina

PLOTS
"""

import matplotlib.pyplot as plt
import numpy as np

x, y = np.loadtxt('Reuvers_n2m50_initial_niter1_T1_xmax50_onlyNORM.txt', delimiter=',', unpack=True)

plt.plot(x,y,'ro')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Schmidt norm after n=50 steps')
plt.legend()
plt.savefig('Reuvers_n2m50_initial_niter1_T1_xmax50_onlyNORM.png')