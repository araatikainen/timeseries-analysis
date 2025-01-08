# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:01:38 2024

@author: turunenj
"""

import numpy as np
import matplotlib.pyplot as plt

time=np.linspace(0,1,1024)          #index time
a=np.random.normal(1,0.05,1024)     #gaussian noise
b=2*time**2                         #Produce nonlinear time series time^2
c=2*time**2.8                       #time series ^2.8                  
 
y=a+b-c;                      #TS+noise
testtime=list(range(0,1024))  #testtime 
testtime2=list(range(1,1100)) #printing time containing small extrapolation section

testtime=np.array(testtime)/100
testtime2=np.array(testtime2)/100

matr=np.zeros((3,1024))     # This should be  3 x 1024 form, transpose it later
matr[0,:]=testtime*0+1
matr[1,:]=testtime**2 #educated quess
matr[2,:]=testtime**3 #educated guess

coeff,resid,rank,s = np.linalg.lstsq(matr.T,y)

y_hat=coeff[0]*(testtime2*0+1)+coeff[1]*(testtime2**2)+coeff[2]*(testtime2**3) #include slight extrapolation
plt.plot(y)
plt.plot(y_hat,'r')
plt.show()