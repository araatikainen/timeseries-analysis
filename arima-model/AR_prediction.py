# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:05:10 2024

@author: turunenj
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import lfilter
from statsmodels.tsa.stattools import levinson_durbin
import sounddevice as sd
import time
Fs,y = wav.read('Kuusi.wav')

y=y-np.mean(y)          #standardixe sound, zero mean, min-max -1...1
y=y/np.abs(np.max(y))

segment=y[3000:3000+512]

plt.plot(segment)
plt.title('Vowel /u/ 512 samples')
plt.show()


#NOTE: concate this snippet to the previous code
[sigma_v, a_coefs, pacf, sigma, phi] =levinson_durbin(segment, nlags=10, isacov=False)
a_coefs=a_coefs*-1 #for some reasons the signs are opposite
 
a_coefs=np.insert(a_coefs,0,1) #inject 1 in to the first place
residual=lfilter(a_coefs,[1],segment);
  
#NOTE: a-coefficients are used as 
#FIR coefficients.
plt.plot(residual)
plt.title('Residual after filtering')
plt.show()
 
offset=55     #first pulse 
# %from beginning
pulse_dist=130-55  #distance between pulses, estimated from figure
amplitude=-0.1 #coarse estimate from figure
# %Note the amplitude levels when compared to %the original speech segment

#Concatenate this to the previous snippets
syn_res=(np.mod(range(1,512),pulse_dist)==0) #let us make a synthetic residual = excitation, add pulse after 
syn_res=syn_res*1                   #true,false to 1,0
syn_res[0]=1                       #first pulse is inserted
syn_res=syn_res*amplitude;          #pulse train scaling
syn_res=np.concatenate((np.zeros((offset)),syn_res))  #inserting offset
syn_res=syn_res[0:512]             #trimming for correct length
syn_res[0]=0.2;                            #not necessarily needed
syn_res=syn_res+np.random.normal(0,0.02,512)    #insert some noise
 
#Plot both residuals original and synthetic
plt.plot(residual)				
plt.plot(syn_res,'r')			   #that will be used as an exitation
plt.title('Original Residual and synthetic residual')
plt.show()

#Concatenate this also 
synt_segm=lfilter([1],a_coefs,syn_res) #prediction, let us synthesize the segment
 
plt.plot(segment)
plt.plot(synt_segm,'r')

#play sounds
sd.play(segment,Fs) #original segment
time.sleep(2)
sd.play(synt_segm,Fs); #synthesized segment
