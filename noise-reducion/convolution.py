# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:36:56 2024

@author: turunenj
"""
import numpy as np
import matplotlib.pyplot as plt

#y=np.convolve(a,v, model='full')
y=np.convolve(1,[3, 2, 1]) #unknown system response by convolution
plt.stem([0, 1, 2],y)
plt.show()

vector=[0, 0, 1, 0, 0, 0.5, 0, 0, 1, 0, 0, 0, 0.5, 0, 0, 0, 1, 0.5, 1, 0.5, 1]
b=[3, 2, 1]
 

plt.stem(vector)
plt.axis([-1, 25, -0.5, 1.5])
plt.title('vector');
plt.show()

plt.stem(b)
plt.axis([-1, 5, -0.5, 4])
plt.title('b')
plt.show()
 
c=np.convolve(b,vector); #convolution
plt.stem(c)
plt.axis([-1, 25, -0.5, 6])
plt.title('c')
plt.show()
