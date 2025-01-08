# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:16:01 2024

@author: turunenj
"""

import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

#triangle parameters
rise_duration = 1
fall_duration = 3
period = 6

control_points_x = [0, rise_duration, rise_duration+fall_duration]
control_points_y = [0, 1, 0]

#define time vector and triangular vector
x = np.linspace(0, 1400, 14300)
triangular_pulses = np.interp(x, control_points_x, control_points_y, period=period)

#compute autocorrelation 
acorr = sm.tsa.acf(triangular_pulses, nlags = len(triangular_pulses)-1)

# Graph
plt.plot(x[0:1000], triangular_pulses[0:1000], label='interpolation');
plt.plot(control_points_x, control_points_y, 'ok', label='control points')
plt.xlabel('x'); plt.ylabel('y'); plt.legend();
plt.title("Triangular pulse train")
plt.show()
#plot full autocorrelation
plt.plot(x[0:14300], acorr[0:14300], label='autocorrelation');
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); 
plt.title("Autocorrelation of triangular pulse train")
plt.show()

#plot parts of it
plt.plot(x[0:1430], acorr[0:1430], label='autocorrelation short sequence');
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()

#add additive gaussian noise
noisy_pulses=triangular_pulses+np.random.normal(0,0.7,len(triangular_pulses))
plt.plot(x[0:1000], noisy_pulses[0:1000]);
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); 
plt.title("Noisy triangular pulse train")
plt.show()

#compute autocorrelation from noisy pulses
acorr_noise = sm.tsa.acf(noisy_pulses, nlags = len(triangular_pulses)-1)
plt.plot(x[0:1430], acorr_noise[0:1430], label='autocorrelation (noisy) short sequence');
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()
