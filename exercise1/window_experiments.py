# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:46:51 2024

@author: turunenj
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
Fs,y=wavfile.read("Kuusi.wav", mmap=False)

win_half_len1=5
win_half_len2=50
x=0*y
x2=0*y
for i in range(0,len(y)):
    print("%d / %d\n" % (i,len(y))) 
    start=i-win_half_len1 
    if start<1: #for initializing the window 
        start=1 
    start2=i-win_half_len2 
    if start2<1: #for initializing the window2 
        start2=1 
    ending=i+win_half_len1 
    if ending>len(y): #taking care of the 
        ending=len(y) #end of the window 
    ending2=i+win_half_len2 
    if ending2>len(y): #taking care of the 
        ending2=len(y) #end of the window2 
    if len(y[start:ending])<2: 
        x[i]=0 
    else: 
        x[i]=np.mean(y[start:ending]) #sliding window mean 
    if len(y[start2:ending2])<2: 
        x2[i]=0 
    else: 
        x2[i]=np.mean(y[start2:ending2]) #sliding window mean 
plt.plot(y[:200],linewidth=0.5) 
plt.plot(x[:200],'r',linewidth=0.5) 
plt.plot(x2[:200],'g',linewidth=0.5) #plot the results 
plt.legend(['Original','3-sample filtered','11-sample filtered']) 
plt.show()