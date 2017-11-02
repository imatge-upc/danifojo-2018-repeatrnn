from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import random

import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input_size = 64
batch = 128
hidden_size = 128
num_layers = 1
num_epochs = 1000
learning_rate = 0.001


def generate():
    x = np.random.randint(3, size=(batch, 1, input_size)) - 1
    y = np.zeros((batch, 1))
    for i in range(batch):
        unique, counts = np.unique(x[i, :, :], return_counts=True)
        try:
            y[i, 0] = dict(zip(unique, counts))[1] % 2
        except:
            y[i, 0] = 0
    x = Variable(torch.from_numpy(x).float())
    y = Variable(torch.from_numpy(y).float())
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    return x, y


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Forward propagate RNN
        h0 = self.init_hidden_state()
        out, h0 = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.sigmoid(out)
        return out

    def init_hidden_state(self):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
        return h0



rnn = RNN(input_size, hidden_size, num_layers)
if torch.cuda.is_available():
    rnn.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def _exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pandas.stats.moments.ewma(values, span=smoothing_factor)


def accuracy(y, out):
    if torch.cuda.is_available():
        y = y.cpu()
        out = out.cpu()
    y = y.data.numpy()
    out = out.data.numpy()
    out = np.round(out)
    return 1 - (sum(abs(y-out))/(batch))\


losses = np.zeros(num_epochs)
acc = np.zeros(num_epochs)
for i in range(num_epochs):
    rnn.zero_grad()
    x, y = generate()
    out = rnn(x)
    loss = criterion(out, y)
    losses[i] = loss.data[0]
    acc[i] = accuracy(out, y)
    loss.backward()
    optimizer.step()
    if (i+1) % (num_epochs//100) == 0:
        np.save(losses, './loss.npy')
        np.save(acc, './acc.npy')
        torch.save(rnn, './rnn')
        print('Step ' + str(i+1) + '/' + str(num_epochs) + ' done. Loss = ' + str(losses[i])
              + '. Accuracy = ' + str(acc[i]))


np.save(losses, './loss.npy')
np.save(acc, './acc.npy')

try:
    plt.plot(losses)
    plt.title('Loss')
    plt.figure()
    plt.plot(acc)
    plt.title('Accuracy')
except:
    print('No display available.')
