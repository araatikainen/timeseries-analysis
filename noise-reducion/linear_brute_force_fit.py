# -*- coding: utf-8 -*-
"""
Created on Thu May 30 08:29:57 2024

@author: turunenj
"""

import numpy as np
import matplotlib.pyplot as plt
#trial and error mechanism, done by brute force search 
sample_size=30;
x=np.arange(0,sample_size,0.02)
ao=0.5 
bo=1.3 #initial, original a and b values for a+bx function
print("Original parameters:")
print("a0=",ao,"b0=", bo) #Let’s print them out
y=ao+bo*x+np.random.normal(0,1.5,len(x)) #add noise
error_sum=99999999  #initial minimum
#brute force seek
for a in np.arange(0.25,2,0.01):    #perform fitting in small steps 
    for b in np.arange(1,5,0.01):
        y1=a+b*x #line equation
        total_sum=np.sum((y-y1)**2) #sum of squares
        #print(a,b,total_sum,error_sum)

        if total_sum<error_sum:  #seek the minimum
            error_sum=total_sum  #and replace good para-     	  
            a1=a                 #meters with better ones
            b1=b
            


a1=np.int32(a1*1000)/1000 #oldie but goldie, better ways are available in Python
b1=np.int32(b1*1000)/1000

print("Estimated[a1 b1] : a1=",a1,"b1=",b1) #Let’s print them out
y1=a1+b1*x  #Make estimation function

plt.plot(x,y,'x'); 
plt.plot(x,y1,'r');
plt.show()
