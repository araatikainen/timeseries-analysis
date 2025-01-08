# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:50:02 2024

@author: turunenj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

name='./Exercise_Train_data.xlsx'

df = pd.read_excel(name,sheet_name='Sheet1') #load data in dataframe
df.drop([0],axis=0, inplace=True) #remove first row

y=df["Column208"]
X1 = df.drop("Column208", axis=1)

for i in [2,10,926]:
    data=X1.iloc[i,:]

    ax1=plt.subplot(311)
    ax1.plot(data[0:69])
    ax1.title.set_text('Standardized heart rate values, index = '+str(i)+', Result = '+str(np.int32(y[i])))
    ax1.set_xticks([])
    ax2=plt.subplot(312)
    ax2.plot(data[69:138])
    ax2.title.set_text('Standardized speed value')
    ax2.set_xticks([])
    ax3=plt.subplot(313)
    ax3.plot(data[138:207])
    ax3.title.set_text('Standardized altitude data')
    ax3.set_xticks([])
    plt.show()