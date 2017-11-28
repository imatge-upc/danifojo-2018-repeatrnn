from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import pandas
from torch.autograd import Variable
import matplotlib.pyplot as plt

from ACT2 import ARNN

batch_size = 10
sequence_length = 4
input_size = 20
hidden_size = 50
output_size = 20
learning_rate = 0.001
steps = 300
lamda = 0


def generate(A):
    x = Variable(torch.randn(input_size))
    y = Variable(torch.zeros(output_size), requires_grad=False)
    y.data = torch.mv(A, x.data)
    return x, y


# Smoothing function for plots.
def _exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pandas.stats.moments.ewma(values, span=smoothing_factor)


arnn = ARNN(input_size, hidden_size, output_size)

optimizer = torch.optim.Adam(arnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
losses = np.zeros(steps)
ponders = np.zeros(steps)
A = 1*torch.randn(output_size, input_size)


def train_step():
    s = Variable(torch.zeros(hidden_size))
    loss = Variable(torch.zeros(1))
    for k in range(sequence_length):
        x, y = generate(A)
        o, s, p = arnn(x, s)
        loss = loss + criterion(o, y) + lamda*p
    return loss, p


for i in range(steps):
    arnn.zero_grad()
    loss = Variable(torch.zeros(1))
    for j in range(batch_size):
        step_loss, p = train_step()
        loss = loss + step_loss
    losses[i] = loss.data[0]
    if i % 10 == 0:
        print('Step ' + str(i) + ' out of ' + str(steps) + '. Loss = ' + str(losses[i]))
    ponders[i] = p.data[0]
    loss.backward()
    optimizer.step()


losses_f = _exponential_moving_average_smoothing(losses, 50)
plt.plot(losses_f)
plt.title('Loss')
plt.figure()
ponders_f = _exponential_moving_average_smoothing(ponders, 50)
plt.plot(ponders_f)
plt.title('Ponder')
plt.show()
