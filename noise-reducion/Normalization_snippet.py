# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:30:32 2024

@author: turunenj
"""

from sklearn.preprocessing import MinMaxScaler,StandardScaler

#scale = MinMaxScaler()
scale = StandardScaler()

# load data


#works with dataframes, modify if necessary
normalization=1 
if normalization==1:
    X_data = scale.fit_transform(data)
else:
    X_data=data.to_numpy()
    
#continue regression/classification