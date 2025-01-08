# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:43:16 2024

@author: turunenj
"""

import numpy as np

def MSE(real_y, pred_y):
    MSE_error=np.mean((np.subtract(real_y,pred_y))**2)
    return MSE_error

def MAE(real_y, pred_y):
    MAE_error=np.mean(np.abs(np.subtract(real_y,pred_y)))
    return MAE_error

def MAPE(real_y, pred_y):
    apu=np.subtract(real_y,pred_y)
    MAPE_error=np.mean(np.abs(np.divide(apu,real_y)))
    #MAPE_error=MAPE_error*100   #For percentages
    return MAPE_error

fs=8000 #sampling frequency Hz 
freq=440 #Hertz 
end_time = 0.1 
time = np.arange(0,end_time,1/fs) # start, stop, step #time series vector 
y=np.sin(2*np.pi*freq*time)+np.random.normal(loc=0.0, scale=0.8, size=[1, len(time)]) #sine curve + added gaussian noise 
y=y.squeeze() #remove unnecessary dimensions

pred_y=0.5*y  #let us attenuate a bit to make a difference

print("MSE error: ",round(MSE(y,pred_y),2))
print("MAE error: ",round(MAE(y,pred_y),2))
print("MAPE error: ",round(MAPE(y,pred_y),2))