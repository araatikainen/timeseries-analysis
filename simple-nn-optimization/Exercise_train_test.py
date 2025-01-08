# -*- coding: utf-8 -*-
import pandas as pd

name='./Exercise_Train_data.xlsx'

df = pd.read_excel(name,sheet_name='Sheet1') #load data in dataframe
df.drop([0],axis=0, inplace=True) #remove first row

y=df["Column208"]
X1 = df.drop("Column208", axis=1)

# now we have input = X and output = y datasets, next we can define 
# the classifier network

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils


#We have to trasform the input and output arrays to 32-bit torch tensors
inputx = torch.tensor(X1.values).float()   #time series vectors
outputy = torch.tensor(y.values).float() #contains the classification data 0,1,2
train_data = data_utils.TensorDataset(inputx, outputy)  # Let us encapsulate the data for
                                                        # easier splitting and shuffling for training
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
                                                        #let us take 5 rows at once, shufflin allowed

#if you have CUDA capable graphics card, it may speed up the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(device)


class classifier_selfmade_network(nn.Module):
    def __init__(
            self,
            inp_units=3*69, #heart rate, speed and altitude data
            num_units=300,   #hidden units
            num_units1=700,   #hidden units
            num_units2=500,   #hidden units
            out_units=3,    #classification 0, 1 or 2
            nonlin=F.leaky_relu,  #this is the activation function that restrict the 
                               #output values of the each neuron in the layer
            nonlin1=F.leaky_relu,  #this is the activation function that restrict the 
                               #output values of the each neuron in the layer
            nonlin2=F.leaky_relu,  #this is the activation function that restrict the 
                               #output values of the each neuron in the layer
    ):
        super(classifier_selfmade_network, self).__init__()
        #introducing the variables to 'self' structure
        self.num_units = num_units
        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.nonlin = nonlin
        self.nonlin1 = nonlin1
        self.nonlin2 = nonlin2
        # the next shows how the data is flowing through the layers
        self.dense0 = nn.Linear(inp_units, num_units)
        self.dense1 = nn.Linear(num_units, num_units)    
        self.dense2 = nn.Linear(num_units, num_units1) 
        self.dense3 = nn.Linear(num_units1, num_units2) 
        self.output = nn.Linear(num_units2, out_units)

    def forward(self, X, **kwargs):         #forward flow
        X = self.nonlin(self.dense0(X))
        X = self.nonlin1(self.dense1(X))
        X = self.nonlin2(self.dense2(X))
        X = self.nonlin1(self.dense3(X))
        X = self.output(X)
        return X
#assign our network to simpler name for future use   
net=classifier_selfmade_network()

#enable paralleilisation for multiple cores or CUDA capable cards automatically
net = nn.DataParallel(net)
#move the network to dedicated device(s)
net.to(device)


#define network basic optimization parameters
import torch.optim as optim

#ADAM (adaptive estimates of lower-order moments) is very good gradient finding algorithm
optimizer=optim.Adam(net.parameters(),lr=0.00125)  #lr=initial learning rate
 
#criterion=nn.MSELoss(reduction='sum') #for regression/prdiction
criterion = nn.CrossEntropyLoss()      # for classification

#scheduler reduces learning rate in places where are no significant gradient sloes in error surface
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9, patience=10, verbose=False)

#let us train the network
for epoch in range(400):
    #logs = {}
    train_loss = 0.0
    # supress Learning Rate after the first epoch
    if epoch>0:
        scheduler.step(loss)
    #go the data trough by the baches
    for (xd,yd) in train_loader:
        yd = yd.type(torch.LongTensor) #for classification problems

        #load the data to the device one batch at a time (input+output)       
        xd = xd.to(device)
        yd = yd.to(device)
        
        #Get predictions from the input values (batch at a time)
        outputti = net(xd)
        #zero the parameter gradients
        optimizer.zero_grad()

        # Calculate Loss:  
        loss = criterion(outputti, yd)
        # Fed the error backwards to the network (learning from mistakes!)
        loss.backward()
        # Updating parameters
        optimizer.step()
        # Collect error for the user
        train_loss += loss.item()

    # Print Learning Rate and temoral epoch error = loss 
    print("Epoch:",epoch, "\tLR:",optimizer.param_groups[0]['lr'],"\tTraining Loss: ", (train_loss / len(train_loader)))   

#testing the newly trained network with test data
import numpy as np
name='./Exercise_Test_data.xlsx'

df_test = pd.read_excel(name,sheet_name='Sheet1') #load data in dataframe
df_test.drop([0],axis=0, inplace=True) #remove first row

yx=df_test["Column208"]
X1_test = df_test.drop("Column208", axis=1)
y_test=yx.to_numpy()
y_test=y_test.astype(int)

tensori=torch.tensor(X1_test.values).float()
input_tensor=tensori.to(device)
test_values=net(input_tensor)
test_values=test_values.detach().cpu().numpy()
correct=0 
total=len(test_values) 

for i in range(0,len(test_values)):
     predicted = np.argmax(test_values[i])
     #print(y_test[i], predicted, test_values[i])
     correct += (predicted == y_test[i]).item()

print("Correct:",correct/total*100, "%")

# Network saving/loading procedure not implemented!!!

