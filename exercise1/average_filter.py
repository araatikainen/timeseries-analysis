#average_filter.py
#Jari Turunen, TUNI
import numpy as np
from numpy import cos, sin, pi, absolute, arange, mean
from matplotlib import pyplot as plt
from scipy.stats import skew,kurtosis
fs=8000     #sampling frequency Hz
freq=440	#Hertz
end_time = 0.1
time = np.arange(0,end_time,1/fs)  # start, stop, step #time series vector
print(len(time))
y=sin(2*pi*freq*time)+np.random.normal(loc=0.0, scale=0.8, size=[1, len(time)]) #sine curve + added gaussian noise
y=y.squeeze()
print(y.shape)

len1=1 #length of the average filter (trend) (1+2*len)=window size
len2=5 #longer trend (1+2*len2)
x=y*0	#fast initialization
x2=y*0
#x3=x2
#x4=x2
for i in range(len(y)):
	print("%d / %d\n" % (i,len(y)))
	start=i-len1
	if start<1: #for initializing the window
		start=1
	start2=i-len2
	if start2<1: #for initializing the window2
		start2=1

	ending=i+len1    
	if ending>len(y):   #taking care of the 
		ending=len(y)	#end of the window

	ending2=i+len2    
	if ending2>len(y):  #taking care of the 
		ending2=len(y) 	#end of the window2

	if len(y[start:ending])<2:
		x[i]=0
	else:
		x[i]=np.mean(y[start:ending])  #sliding window mean
	if len(y[start2:ending2])<2:
		x2[i]=0
	else:
		x2[i]=np.mean(y[start2:ending2]) #sliding window mean
		#x3[i]=skew(y[start2:ending2],axis=0, bias=True)
		#x4[i]=kurtosis(y[start2:ending2],axis=0, bias=True)



plt.plot(y)
plt.plot(x,'r')
plt.plot(x2,'g') #plot the results
#plt.plot(x3,'b')
#plt.plot(x4,'y') #plot the results
plt.legend(['Original','3-sample filtered','11-sample filtered'])
plt.show()