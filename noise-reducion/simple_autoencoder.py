# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:51:01 2024

@author: turunenj
"""
#https://stackoverflow.com/questions/78612519/how-to-create-the-denoise-autoencoder-model-with-the-pytorch-for-the-time-series

import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import trim_mean
import matplotlib.pyplot as plt

#Stacked Denoising AutoEncoder = SDAE


# 1.Dataset, I only use the 'capacity' column.
#df = pd.read_csv("Final-CS2_35.csv")
df = pd.read_csv("../CS2_35.csv")
# print(df.shape)  # 882*6
split = 0.8  
r = np.random.normal(0, 0.25, df['capacity'].shape)  
#data = torch.from_numpy(df['capacity'].values).reshape(-1, 1).float() #orignal data
#data_res = torch.from_numpy(df['capacity'].values + r).reshape(-1, 1).float() #train data = original + noise (r) 
data=df['capacity'].values
print(data.shape)
#data_res=df['capacity'].values + r


#make some test_data with different noise
r = np.random.normal(0, 0.25, df['capacity'].shape)  
data_res_test = df['capacity'].values + r #test data = original + noise (r) 


train_len = int(len(data)*split)
print(train_len)
train_data_orig=data[0:train_len]
val_data_orig=data[train_len+1:]
input_size=100

#Let us divide the training data to clean and noisy data

train_data1=np.zeros((len(train_data_orig)-input_size,input_size))
print(train_data1.shape)
for i in range(0,len(train_data_orig)-input_size):
    train_data1[i,:]=train_data_orig[i:i+input_size]
    
val_data=np.zeros((len(val_data_orig)-input_size,input_size))
print(val_data.shape)
for i in range(0,len(val_data_orig)-input_size):
    val_data[i,:]=val_data_orig[i:i+input_size]

#Let us make test data using different noise
r = np.random.normal(0, 1, [len(train_data_orig)])  
train_data_noisy_orig=train_data_orig+r
r = np.random.normal(0, 1, val_data_orig.shape)  
val_data_noisy_orig=val_data_orig+r    

train_data_noisy=np.zeros((len(train_data_noisy_orig)-input_size,input_size))
print(train_data_noisy_orig.shape)
for i in range(0,len(train_data_orig)-input_size):
    train_data_noisy[i,:]=train_data_noisy_orig[i:i+input_size]
    
val_data_noisy=np.zeros((len(val_data_noisy_orig)-input_size,input_size))
print(val_data_noisy.shape)
for i in range(0,len(val_data_orig)-input_size):
    val_data_noisy[i,:]=val_data_noisy_orig[i:i+input_size]

#Let us leave the test data as it is, we must reconstruct it afterwards
test_data1=np.zeros((len(data_res_test)-input_size,input_size))
for i in range(0,len(data_res_test)-input_size):
    test_data1[i,:]=data_res_test[i:i+input_size]


#tensor 

train_data_clean=torch.from_numpy(train_data1).float()
val_data_clean=torch.from_numpy(val_data).float()
train_data_noisy=torch.from_numpy(train_data_noisy).float()
val_data_noisy=torch.from_numpy(val_data_noisy).float()
test_data=torch.from_numpy(test_data1).float()


# 2. create Model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()             #hourglass shaped encoder+decoder combo
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.Tanh(),                
            nn.Linear(hidden_size * 2, hidden_size))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.Tanh(),           
            nn.Linear(hidden_size*2, input_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# 3.train the Model
def train_ae(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for x_data,y_data in train_loader:   #noisy data in x, clean data in y
            optimizer.zero_grad()
            outputs = model(x_data)
            loss = criterion(outputs, y_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_data,y_data in val_loader:   #noisy data in x, clean data in y
                outputs = model(x_data)
                loss = criterion(outputs, y_data)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
    #save model
    torch.save(model.state_dict(), "one-DAE.pkl")

# DataLoader
batch_size = 10

train_data_comb = TensorDataset(train_data_noisy, train_data_clean)
val_data_comb = TensorDataset(val_data_noisy, val_data_clean)
train_loader = DataLoader(train_data_comb, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_comb, batch_size=batch_size, shuffle=False)

ae1 = Autoencoder(input_size=input_size, hidden_size=64)
optimizer = torch.optim.Adam(ae1.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_ae(ae1, train_loader, val_loader, 100, criterion, optimizer)

# 4.predict
def make_predictions_from_dataloader(model, unshuffled_dataloader):
    #load model for evaluation
    model.load_state_dict(torch.load("one-DAE.pkl"))
    model.eval()
    predictions = []
    for x in unshuffled_dataloader:
        with torch.no_grad():
          p = model(x)
          predictions.append(p)
    predictions = torch.cat(predictions).numpy()
    return predictions.squeeze()


res_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
P = make_predictions_from_dataloader(ae1, res_loader)

# 5.Plot
fig = plt.figure(figsize=(16, 9))
plt.gca().set_xticks([]) #remove global xticks
plt.xlim(left=0, right=900)
plt.ylim(bottom=0, top=1.2)

plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
font = {'family': 'Times New Roman', 'size': 16, 'weight': 'normal',}
#matplotlib.rc('font', **font)
plt.rc('font', **font)

print(P.shape)

plt.subplot(3, 1, 1)
plt.plot(df['cycle'][:676], data[:676], linewidth=1.5, color='b')
plt.title('pure time series', fontname='Times New Roman', fontsize=20)

plt.subplot(3, 1, 2)
plt.plot(df['cycle'][:676], train_data_noisy_orig[:676], linewidth=1.5, color='r')
plt.title('Noise-added time series', fontname='Times New Roman', fontsize=20)

#this is a quick and dirty reconstruction, it should be done using sliding window basis to each slot
#however this is nearly sufficient in this case although the amplitude needs further adjustment, especially 
#at the end of the time series
#mean_P=trim_mean(P,0.2,axis=1)

mean_P=np.zeros([len(test_data1),len(test_data1)+input_size],dtype='float32')
m,n=P.shape
print(P.shape)
counter=0
for i in range(0,m):
    mean_P[i,counter:counter+input_size]=P[i,:]
    counter+=1
m,n=mean_P.shape
print(mean_P.shape)
test_vector=np.zeros([n,],dtype='float32')
for i in range(0,n):
    vector=np.trim_zeros(mean_P[:,i], trim='fb')
    test_vector[i]=np.mean(vector)


#The curve shape is quite clean, but the underlying noise is affecting to the baseline amplitude
#Perhaps bigger and deeper network will adjust the base level behaviour???
#you have to test it...

plt.subplot(3, 1, 3)
plt.plot(df['cycle'][:676], test_vector[:676], linewidth=1.5, color='g')
plt.title('SDAE cleaned time series', fontname='Times New Roman', fontsize=20)

plt.show()