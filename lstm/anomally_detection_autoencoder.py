# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:54:51 2024

@author: turunenj
"""
#https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/


#!pip install -qq arff2pandas
#!pip install -q -U watermark
#!pip install -qq -U pandas

#%reload_ext watermark
#%watermark -v -p numpy,pandas,torch,arff2pandas

###Load data

#!gdown --id 16MIleqoIr1vYxlGk4GKnGmrsCPuWkkpT
#!unzip -qq ECG5000.zip

import torch
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
from arff2pandas import load


#%matplotlib inline                             #uncomment for jupyter/colab
#%config InlineBackend.figure_format='retina'   #uncomment for jupyter/colab

#Definitions
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#The data comes in multiple formats. We'll load the arff files into Pandas data frames:
with open('./ECG5000/ECG5000_TRAIN.arff') as f:
  train = load(f)
  
with open('./ECG5000/ECG5000_TEST.arff') as f:
  test = load(f)
#We'll combine the training and test data into a single data frame. This will give us more data to train our Autoencoder. We'll also shuffle it:
#df = train.append(test) #not working an ymore
df=pd.concat([train,test],ignore_index=True)

df = df.sample(frac=1.0)

print(df.shape)
print(df.head())

#We have 5,000 examples. Each row represents a single heartbeat record. Let's name the possible classes:
CLASS_NORMAL = 1
class_names = ['Normal','R on T','PVC','SP','UB']

#Next, we'll rename the last column to target, so its easier to reference it:
new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns

###Exploratory Data Analysis
#Let's check how many examples for each heartbeat class do we have:
print(df.target.value_counts())

#Let's plot the results:
ax = sns.countplot(df.target)
#seaborn sometimes make horizontal bars sometime vertical
#based on df.target orientation, change the next rows, if necessary:

#ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
plt.show()
#The normal class, has by far, the most examples. This is great because we'll use it to train our model.

#Let's have a look at an averaged (smoothed out with one standard deviation on top and bottom of it) Time Series for each class:
def plot_time_series_class(data, class_name, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.125
  )
  ax.set_title(class_name)

classes = df.target.unique()

fig, axs = plt.subplots(
  nrows=len(classes) // 3 + 1,
  ncols=3,
  sharey=True,
  figsize=(14, 8)
)

for i, cls in enumerate(classes):
  ax = axs.flat[i]
  data = df[df.target == cls] \
    .drop(labels='target', axis=1) \
    .mean(axis=0) \
    .to_numpy()
  plot_time_series_class(data, class_names[i], ax)

fig.delaxes(axs.flat[-1])
fig.tight_layout()

###LSTM Autoencoder

#The Autoencoder's job is to get some input data, pass it through the model, and obtain a reconstruction of the input. 
#The reconstruction should match the input as much as possible. The trick is to use a small number of parameters, 
#so your model learns a compressed representation of the data.

#In a sense, Autoencoders try to learn only the most important features (compressed version) of the data. 
#Here, we'll have a look at how to feed Time Series data to an Autoencoder. We'll use a couple of LSTM layers 
#(hence the LSTM Autoencoder) to capture the temporal dependencies of the data.

#To classify a sequence as normal or an anomaly, we'll pick a threshold above which a heartbeat is considered abnormal.
#Reconstruction Loss

#When training an Autoencoder, the objective is to reconstruct the input as best as possible. 
#This is done by minimizing a loss function (just like in supervised learning). 
#This function is known as reconstruction loss. Cross-entropy loss and Mean squared error are common examples.

#Anomaly Detection in ECG Data

#We'll use normal heartbeats as training data for our model and record the reconstruction loss. But first, 
#we need to prepare the data:
    
#Data Preprocessing

#Let's get all normal heartbeats and drop the target (class) column:
    
normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
print(normal_df.shape)
#We'll merge all other classes and mark them as anomalies:
anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)
print(anomaly_df.shape)

#We'll split the normal examples into train, validation and test sets:
train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33, 
  random_state=RANDOM_SEED
)
    
#We need to convert our examples into tensors, so we can use them to train our Autoencoder. 
#Let's write a helper function for that:

def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features



#Each Time Series will be converted to a 2D Tensor in the shape sequence length x number of features (140x1 in our case).

#Let's create some datasets:
    
train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

#The general Autoencoder architecture consists of two components. An Encoder that compresses the input and 
#a Decoder that tries to reconstruct it.

#We'll use the LSTM Autoencoder from this GitHub repo with some small tweaks. 
#Our model's job is to reconstruct Time Series data. Let's start with the Encoder:
class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
      bidirectional=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim *2,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

    #self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
    #self.relu = nn.ReLU()

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    #hidden_n = self.linear(hidden_n)
    #hidden_n = self.relu(hidden_n)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

#The Encoder uses two LSTM layers to compress the Time Series data input.
#Next, we'll decode the compressed representation using a Decoder:

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True,
      bidirectional=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim *2,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
    )

    self.output_layer = nn.Linear(self.hidden_dim,n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)



#Our Decoder contains two LSTM layers and an output layer that gives the final reconstruction.
#Time to wrap everything into an easy to use module:
class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

#Our Autoencoder passes the input through the Encoder and Decoder. Let's create an instance of it:
model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)

def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in tqdm(train_dataset, desc=f"Epoch {epoch} Training", unit="batch"):
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history

#At each epoch, the training process feeds our model with all training examples and evaluates 
#the performance on the validation set. Note that we're using a batch size of 1 
#(our model sees only 1 sequence at a time). We also record the training and validation set losses during the process.

#Note that we're minimizing the L1Loss, which measures the MAE (mean absolute error). 
#Why? The reconstructions seem to be better than with MSE (mean squared error).

#We'll get the version of the model with the smallest validation error. Let's do some training:
    
model, history = train_model(
  model, 
  train_dataset, 
  val_dataset, 
  n_epochs=150
)

#Then plot the learning curves:
    
ax = plt.figure().gca()

ax.plot(history['train'])
ax.plot(history['val'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'])
plt.title('Loss over training epochs')
plt.show()

#Let's store the model for later use:
MODEL_PATH = 'model.pth'

torch.save(model, MODEL_PATH)

#With our model at hand, we can have a look at the reconstruction error on the training set. 
#Let's start by writing a helper function to get predictions from our model:

def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

#Our function goes through each example in the dataset and records the predictions and losses. 
#Let's get the losses and have a look at them:
    
_, losses = predict(model, train_dataset)

sns.displot(losses, bins=50, kde=True)
#Lets us make athreshold based on previous image
THRESHOLD = 26

#Evaluation
#Using the threshold, we can turn the problem into a simple binary classification task:

#    If the reconstruction loss for an example is below the threshold, we'll classify it as a normal heartbeat
#    Alternatively, if the loss is higher than the threshold, we'll classify it as an anomaly

#Normal hearbeats

#Let's check how well our model does on normal heartbeats. 
#We'll use the normal heartbeats from the test set (our model haven't seen those):
predictions, pred_losses = predict(model, test_normal_dataset)
sns.displot(pred_losses, bins=50, kde=True)
#We'll count the correct predictions:

correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

#Anomalies
#We'll do the same with the anomaly examples, but their number is much higher. 
#We'll get a subset that has the same size as the normal heartbeats:
anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
predictions, pred_losses = predict(model, anomaly_dataset)
sns.displot(pred_losses, bins=50, kde=True)

#Finally, we can count the number of examples above the threshold (considered as anomalies):
correct = sum(l > THRESHOLD for l in pred_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')

#Looking at Examples
#We can overlay the real and reconstructed Time Series values to see how close they are. 
#We'll do it for some normal and anomaly cases:
def plot_prediction(data, model, title, ax):
  predictions, pred_losses = predict(model, [data])

  ax.plot(data, label='true')
  ax.plot(predictions[0], label='reconstructed')
  ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
  ax.legend()

fig, axs = plt.subplots(
  nrows=2,
  ncols=6,
  sharey=True,
  sharex=True,
  figsize=(22, 8)
)

for i, data in enumerate(test_normal_dataset[:6]):
  plot_prediction(data, model, title='Normal', ax=axs[0, i])

for i, data in enumerate(test_anomaly_dataset[:6]):
  plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

fig.tight_layout()
