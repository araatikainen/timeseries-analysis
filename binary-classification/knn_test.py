# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:16:49 2024

@author: turunenj
"""

#https://realpython.com/k-means-clustering-python/
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


#generate datasets (3 classes) with old school style
#for more sophisticated way look at make_blobs
num1=400
num2=800
num3=600

data_x1=np.random.normal(3,0.125,num1)
data_y1=np.random.normal(3,0.125,num1)
data_z1=np.ones((num1)) #index class=1

data_x2=np.random.normal(4,0.25,num2)
data_y2=np.random.normal(4,0.25,num2)
data_z2=np.ones((num2))*2 #index class=2

data_x3=np.random.normal(3,0.125,num3)
data_y3=np.random.normal(4,0.25,num3)
data_z3=np.ones((num3))*3 #index class=3

data_x=np.concatenate((data_x1,data_x2,data_x3))
data_y=np.concatenate((data_y1,data_y2,data_y3))
data_z=np.concatenate((data_z1,data_z2,data_z3))

print(data_x.shape)
print(data_y.shape)
print(data_z.shape)
#arrange it for feeding it to knn
data=np.zeros((2,len(data_x)))
data[0,:]=data_x
data[1,:]=data_y

data=data.T  #(2x1800)

##standardization not implemented for visualization reasons
##Standardscaler produces 0 mean, 1 std data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


plt.plot(data_x1,data_y1,'bx')
plt.plot(data_x2,data_y2,'rx')
plt.plot(data_x3,data_y3,'gx')
plt.title('Data')
plt.show()

#How do we know the number of clusters in unknown data?
#Let us test silhouette algorithm

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []
#Notice you start at 2 clusters to 11 for silhouette coefficient
for k in range(2, 11):
      kmeans = KMeans(n_clusters=k)
      kmeans.fit(scaled_data) #change scaled_data here
      score = silhouette_score(data, kmeans.labels_)
      silhouette_coefficients.append(score)
      
      
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
#Based on silhouette image let us take the maximum = 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

print(kmeans.cluster_centers_)

plt.plot(data_x1,data_y1,'bx')
plt.plot(data_x2,data_y2,'rx')
plt.plot(data_x3,data_y3,'gx')
plt.plot(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1],'kx')
plt.plot(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1],'kx')
plt.plot(kmeans.cluster_centers_[2,0],kmeans.cluster_centers_[2,1],'kx')
plt.title('Data with cluster centers')
plt.show()