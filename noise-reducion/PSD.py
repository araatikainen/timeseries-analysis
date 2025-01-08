# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:38:36 2024

@author: turunenj
"""

from scipy.io import wavfile 
import numpy as np 
from scipy import signal 
import matplotlib.pyplot as plt 

from scipy.io import wavfile 
import numpy as np 
from scipy import signal 
import matplotlib.pyplot as plt 
samplerate, x = wavfile.read('./kuusi.wav') 
start=2600 
ending=3000 #test different parts of the time series 
y=np.fft.fft(x[start:ending]) #compute FFT 
y=np.abs(y) 
y=y[1:np.int32(len(y)/2)] #take half of it 
f,PSDz=signal.welch(x[start:ending],samplerate,scaling='density') #compute PSD 
freqq=np.linspace(1,np.ceil(samplerate/2),len(y)) #map indices to frequencies 
freq_z=np.linspace(1,np.ceil(samplerate/2),len(PSDz)) #map indices to frequencies for PSD 
fig, axs = plt.subplots(4,1,figsize=(10,10)) 
fig.suptitle('FFT vs. PSD') 
axs[0].plot(x[start:ending]) 
axs[0].set_title("Original") 
axs[1].plot(freqq,y) 
axs[1].set_title("abs(FFT)") 
axs[2].semilogy(f, PSDz) 
axs[2].set_title("PSD") 
axs[3].plot(f, 10*np.log10(PSDz)) 
axs[3].set_ylim([-40,20]) 
axs[3].set_title("PSD in dB") 
fig.tight_layout() 
plt.show()
