from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import ACT

input_size = 64
batch = 128
hidden_size = 128
num_layers = 1
num_epochs = 10000
learning_rate = 0.001
lamda = 1e-5


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
    y = y.view(-1)
    return x, y


class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_first=True):
        super(ARNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.arnn = ACT.ARNN(input_size, hidden_size, output_size, num_layers, batch_first=True)

    def forward(self, x):
        # Forward propagate RNN
        if self.batch_first:
            batch = x.size()[0]
        else:
            batch = x.size[1]
        h0 = self.init_hidden_state(batch)
        out, p = self.arnn(x, h0)
        out = F.sigmoid(out)
        return out, p

    def init_hidden_state(self, batch):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
        return h0


arnn = ARNN(input_size, hidden_size, num_layers)
if torch.cuda.is_available():
    arnn.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(arnn.parameters(), lr=learning_rate)


def _exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pandas.stats.moments.ewma(values, span=smoothing_factor)


def accuracy(y, out):
    y = y.data.view(-1)
    out = out.data.view(-1).round()
    return 1 - torch.sum(torch.abs(y-out))/(batch)


losses = np.zeros(num_epochs)
acc = np.zeros(num_epochs)
ponders = np.zeros(num_epochs)
for i in range(num_epochs):
    arnn.zero_grad()
    x, y = generate()
    out, p = arnn(x)
    p_sum = torch.sum(p)
    out = out.view(-1)
    loss = criterion(out, y) + lamda*p_sum
    losses[i] = loss.data[0]
    ponders[i] = p_sum.data[0]
    acc[i] = accuracy(out, y)
    loss.backward()
    optimizer.step()
    if (i+1) % (num_epochs//100) == 0:
        np.save('./results/parity_loss.npy', losses)
        np.save('./results/parity_accuracy.npy', acc)
        np.save('./results/parity_ponder.npy', ponders)
        torch.save(arnn.state_dict(), './results/parity_model.torch')
        print('Step ' + str(i+1) + '/' + str(num_epochs) + ' done. Loss = ' + str(losses[i])
              + '. Accuracy = ' + str(acc[i]))


np.save('./results/parity_loss.npy', losses)
np.save('./results/parity_accuracy.npy', acc)
np.save('./results/parity_ponder.npy', ponders)
torch.save(arnn.state_dict(), './results/parity_model.torch')

try:
    losses_f = _exponential_moving_average_smoothing(losses, 51)
    plt.plot(losses_f)
    plt.title('Loss')
    plt.figure()
    acc_f = _exponential_moving_average_smoothing(acc, 51)
    plt.plot(acc_f)
    plt.title('Accuracy')
    plt.figure()
    ponders_f = _exponential_moving_average_smoothing(ponders, 51)
    plt.plot(ponders_f)
    plt.title('Ponder')
    plt.show()
except:
    print('No display available.')
