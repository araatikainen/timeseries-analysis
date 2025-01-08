# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:29:15 2024

@author: turunenj
"""
import numpy as np
import matplotlib.pyplot as plt

time=np.linspace(0,1,1024)          #index time
a=np.random.normal(1,0.05,1024)     #gaussian noise
b=time**2                           #Produce nonlinear time series time^2
 
y=a+b                               #Time Series+noise
testtime=list(range(0,1024))        #test time
testtime2=list(range(1,1100))       #printing time

#testtime=np.transpose(np.array(testtime))
#testtime2=np.transpose(np.array(testtime2))
testtime=np.array(testtime)
testtime2=np.array(testtime2)

matr=np.zeros((2,1024)) 
matr[0,:]=testtime*0+1
matr[1,:]=testtime**2 #let us make test matrix 
                                 #for constant and second power function
coeff,resid,rank,s = np.linalg.lstsq(matr.T,y) #fit the matrix with respect to 
                                               #y to obtain coefficients, 
                                               #note: left division operation
 
y_hat=coeff[0]*(testtime2*0+1)+coeff[1]*(testtime2**2)
                            #extrapolate function with
 		        	        #longer time
plt.plot(y)
plt.plot(y_hat,'r')
plt.show()