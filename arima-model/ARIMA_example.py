# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:10:47 2024

@author: turunenj
"""

# Please look at these original links

# https://www.projectpro.io/article/how-to-build-arima-model-in-python/544

# https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html

# for more instructions and explanations

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.api import qqplot
import pandas as pd

#define z-score for partial autocorrelation analysis
#https://en.wikipedia.org/wiki/Standard_score

z_score=3 #equal to 3-sigma i.e. ~99% of the data
len_segment=512
#https://en.wikipedia.org/wiki/Partial_autocorrelation_function
reference=z_score/np.sqrt(len_segment)



Fs,y = wav.read('Kuusi.wav')

y=y-np.mean(y)          #standardixe sound, zero mean, min-max -1...1
y=y/np.abs(np.max(y))

segment=y[3000:3000+len_segment] #take a vowel /u/ segment
test_segment=y[3000+len_segment+1:3600]


plt.plot(segment)
plt.title('Vowel /u/ 512 samples')
plt.show()

#show autocorrelation and partial autocorrelation of the segment
#Look at the place of first change of sign in the images
print("Let us define the ARIMA(p, d, q) parameters")

print("\n*** Look at the biggest index (lag) from partial autocorrelation (pacf) image that is still larger than reference value")
print("Select the p = order (lag) reference value +1 ***")

print("q can be estimated similarly by looking at the ACF plot instead of the PACF plot.")
print("Looking at the number of lags crossing the threshold, we can determine how much of the past")
print("would be significant enough to consider for the future.")
print("The ones with high correlation contribute more and would be enough to predict future values")

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(segment.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(segment, lags=40, ax=ax2)
ax2.axhline(y=reference, color='r')
ax2.axhline(y=-reference, color='r')
plt.show()

mod1 = ar_select_order(segment, maxlag=15,ic='bic')
mod2 = ar_select_order(segment, maxlag=15,ic='aic', seasonal=True, period=10)
mod3 = ar_select_order(segment, maxlag=15,ic='hqic', glob=True)
#the problem is that even AR selection will produce an array of lag values
#and not a single good number
#however they contain the same values as the AC and partial AC sign shange indices
print("\nPrint AR model order estimation lags (not much help) let us stick on the pacf lag+1 = p")
print(mod1.ar_lags)
print(mod2.ar_lags)
print(mod3.ar_lags)

#https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
print("\nLet us find the d (differencing) value, the first one that goes below 0.05 limit:")
result = adfuller(segment)
print('p-value (0.05): ', result[1])
result = adfuller(np.diff(segment))
print('1st order diff p-value (0.05): ', result[1])
result = adfuller(np.diff(np.diff(segment)))
print('2nd order diff p-value (0.05): ', result[1])
result = adfuller(np.diff(np.diff(np.diff(segment))))
print('3rd order diff p-value (0.05): ', result[1])
 
arma_model=ARIMA(segment,order=(5,0,5)).fit()
print("\n\nModel summary")
print(arma_model.summary()) 

print(arma_model.params)
print("\nPrint AIC, BIC, HQIC values for full autocorrelation")
print(arma_model.aic, arma_model.bic, arma_model.hqic)

#print residual values, we can estimate the qoodness of the fit by analyzing the residuals
#The closer to 0 the statistic, the more evidence for positive serial correlation. 
#The closer to 4, the more evidence for negative serial correlation.
#https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html
print("\nPrint residual values")
print(sm.stats.durbin_watson(arma_model.resid))

fig = plt.figure(figsize=(12, 8))
plt.plot(arma_model.resid,label='residual of the model')
plt.legend()
plt.show()

print("\nResidual normalization test for both")
print("Full")
print(stats.normaltest(arma_model.resid))

print("\nResidual against normal distribution comparison image")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(211)
fig = qqplot(arma_model.resid, line="q", ax=ax, fit=True)



#Full residual
print("Residual autocorrelation analysis")
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_model.resid.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_model.resid, lags=40, ax=ax2)
plt.show()

#Organize data to neat datatable format
r, q, p = sm.tsa.acf(arma_model.resid.squeeze(), fft=True, qstat=True)
data = np.c_[np.arange(1, 28), r[1:], q, p]

table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
print("\nDataFrame")
print(table.set_index("lag"))


#finally, predictions from model

predict_vowel = arma_model.predict(513, 600, dynamic=True) 
fig = plt.figure(figsize=(12, 8))
plt.plot(test_segment,label='Original')
plt.plot(predict_vowel,'r', label='Prediction')
plt.title("Original and predicted")
plt.legend()
plt.show()

