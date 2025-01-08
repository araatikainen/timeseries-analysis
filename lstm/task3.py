# %% [markdown]
# ### Exercise 5

# %% [markdown]
# #### Task 3

# %% [markdown]
# Follow the example from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

# %% [markdown]
# Import libraries

# %%
import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Load seaborn "flights" dataset

# %%
flight_data = sns.load_dataset("flights")
print(flight_data.shape)
flight_data.head()

# %% [markdown]
# Plot frequency of passengers per month.

# %%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(flight_data['passengers'])

# %% [markdown]
# Preprocess data and split into train and test sets

# %%
all_data = flight_data['passengers'].values.astype(float)

test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

# %% [markdown]
# Normalize data using min max scaler.

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

# %% [markdown]
# Convert dataset into tensors.

# %%
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# %% [markdown]
# define a function named create_inout_sequences. The function will accept the raw input data and will return a list of tuples.

# %%
train_window = 12

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

# %% [markdown]
# Create LSTM model

# %%
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        #self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers=2, batch_first=True)
        #self.gru = nn.GRU(input_size, hidden_layer_size, num_layers=2, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        #self.hidden_cell = torch.zeros(2,1,self.hidden_layer_size) # RNN/GRU

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        #rnn_out, self.hidden_cell = self.rnn(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        #gru_out, self.hidden_cell = self.gru(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# %% [markdown]
# Define loss function and optimizer

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM().to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# Train the model


# %%
epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        seq = seq.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))
        #model.hidden_cell = torch.zeros(2, len(seq), model.hidden_layer_size).to(device) # RNN/GRU
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# %% [markdown]
# Make predictions

# %%
fut_pred = 12

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)

model.eval()

model.hidden = (torch.zeros(2, 1, model.hidden_layer_size),
                torch.zeros(2, 1, model.hidden_layer_size))
#model.hidden = torch.zeros(2, fut_pred, model.hidden_layer_size) # RNN/GRU
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
    with torch.no_grad():
        test_inputs.append(model(seq).item())

test_inputs[fut_pred:]

# %% [markdown]
# Convert into actual values using inverse_transform

# %%
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)

# mse
from sklearn.metrics import mean_squared_error
print('Mean Squared Error:', mean_squared_error(test_data, actual_predictions))
print(test_data)

# %%
x = np.arange(132, 144, 1)

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x,actual_predictions)
plt.show()

# %%
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(flight_data['passengers'][-train_window:])
plt.plot(x,actual_predictions)
plt.show()

# %% [markdown]
# Model is now predicting more accurate results and is able to capture seasonal change. The first plot shows in the orange colour that the line mimics the actual data. This is better than the original which is below. 
# 
# Two linear layers (pre_linear in the code) with size of 64 was added to the architecture before the lstm layer. Second pre linear layer returns the size to the size of input for lstm. Hidden_layer_size was increased from 100 to 400, which was tested by trial and error. Higher values are not able to catch the peak of the trend.
# 
# Results with original model:
# 
# ![image.png](attachment:image.png)
# image from https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf
