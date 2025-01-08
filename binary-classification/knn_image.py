# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:16:49 2024

@author: turunenj
"""

#https://realpython.com/k-means-clustering-python/
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


features, true_labels = make_blobs(
   n_samples=200,
   centers=3,
   cluster_std=2.75,
   random_state=42
   )

#generate datasets (3 classes)
num1=400
num2=800
num3=600

data_x1=np.random.normal(3,0.125,num1)
data_y1=np.random.normal(3,0.125,num1)
data_z1=np.ones((1,num1)) #index class=1

data_x2=np.random.normal(4,0.25,num2)
data_y2=np.random.normal(4,0.25,num2)
data_z2=np.ones((1,num2))*2 #index class=2

data_x3=np.random.normal(3,0.125,num3)
data_y3=np.random.normal(4,0.25,num3)
data_z3=np.ones((1,num3))*3 #index class=3

centroid_x1=np.random.normal(3,0.5,1)
centroid_y1=np.random.normal(3,0.5,1)

centroid_x2=np.random.normal(4,0.5,1)
centroid_y2=np.random.normal(4,0.5,1)

centroid_x3=np.random.normal(3,0.5,1)
centroid_y3=np.random.normal(4,0.5,1)

plt.plot(data_x1,data_y1,'bx')
plt.plot(data_x2,data_y2,'rx')
plt.plot(data_x3,data_y3,'gx')
plt.plot(centroid_x1,centroid_y1,'kx')
plt.plot(centroid_x2,centroid_y2,'kx')
plt.plot(centroid_x3,centroid_y3,'kx')

plt.show()

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []
#Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
     kmeans.fit(scaled_features)
     score = silhouette_score(scaled_features, kmeans.labels_)
     silhouette_coefficients.append(score)