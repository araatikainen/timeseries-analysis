# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:16:01 2024

@author: turunenj
"""

import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm


#define time vector and sine vector
Fs=2000 #sampling frequency
x = np.linspace(0, 1, Fs)
sin_combo = np.sin(x*2*np.pi*630)+np.sin(x*2*np.pi*940) #630Hz and 940Hz

#compute autocorrelation 
acorr = sm.tsa.acf(sin_combo, nlags = len(sin_combo)-1)

# Graph
plt.plot(x[0:500], sin_combo[0:500]);
plt.xlabel('x [s]'); plt.ylabel('y'); plt.legend();
plt.title("Sine 630Hz+940Hz")
plt.show()
#plot full autocorrelation
plt.plot(x, acorr, label='autocorrelation');
plt.xlabel('x [s]'); plt.ylabel('y'); plt.legend(); 
plt.title("Autocorrelation of sine train")
plt.show()

#plot partial autocorrelation
plt.plot(x[0:200], acorr[0:200], label='autocorrelation');
plt.xlabel('x [s]'); plt.ylabel('y'); plt.legend(); 
plt.title("Partial Autocorrelation of sine train")
plt.show()
