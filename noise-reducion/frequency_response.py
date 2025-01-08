# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:16:18 2024

@author: turunenj
"""
import numpy as np
from scipy.signal import lfilter,freqz
import matplotlib.pyplot as plt

x=np.random.normal(0.0, 0.01, 1024)
b=[0.33, 0.33, 0.33] #3-sample mean filter 
y=lfilter(b,[1],x)     #actual filtering operation
#plotting

plt.plot(x)		# plot data
plt.plot(y,'r')	# 
plt.legend(['x','y']);
plt.show()


X=np.fft.fft(x); #frequency conversion for input
Y=np.fft.fft(y); #frequency conversion for output 
H=np.divide(Y,X)   #frequency response computation
Hb=freqz(b,1,512); #freqz function models the b-coefficients frequency response using polynomial function 
#NOTE, x-axle frequencies are just indices, not real frequencies
#indices must be scaled for real frequencies, if needed
plt.plot(np.abs(H))
plt.plot(np.abs(Hb[1]),'r')

plt.legend(['Signal based frequency response','Freqz-based frequency response'])
plt.show()

h1=np.fft.ifft(H); #inverse frequency conversion for output 
print(np.abs(h1[0:10]))

