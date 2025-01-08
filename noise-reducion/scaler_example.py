# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:41:53 2024

@author: turunenj
"""


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#data must be greater than one dimension in columnwise
data=[[0,1],
      [2,3],
      [4,5],
      [6,7],
      [8,9],
      [10,11]]

print('Data')
print(data)
scalerMM = MinMaxScaler()

scaled_minmax = scalerMM.fit_transform(data)
print('\nScaled minmax')
print(scaled_minmax)

scalerS = StandardScaler()
scaled_s = scalerS.fit_transform(data)
print('\nScaled standard')
print(scaled_s)

#perform inverse transform
inverted_minmax=scalerMM.inverse_transform(scaled_minmax)
print('\nInverted minmax')
print(inverted_minmax)


inverted_s=scalerS.inverse_transform(scaled_s)
print('\nInverted standard')
print(inverted_s)