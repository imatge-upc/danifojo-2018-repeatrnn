from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tqdm import trange
import pandas as pd

from ACT import SkipARNN as ARNN

batch_size = 10
sequence_length = 5
input_size = 20
hidden_size = 64
output_size = 20
learning_rate = 0.0001
steps = 250
tau = 0.1
num_processes = 8


def generate(A, b):
    x = Variable(torch.randn(input_size))
    y = Variable(torch.zeros(output_size), requires_grad=False)
    y.data = torch.mv(A, x.data) + b
    return x, y


def train_loop(rank, model, criterion, optimizer, tau, q):
    if rank == 0:
        pbar = trange(steps)
        losses = []
        ponders = []
    else:
        pbar = range(steps)
    for i in pbar:
        model.zero_grad()
        loss = Variable(torch.zeros(1))
        p_sum = Variable(torch.zeros(1))
        for j in range(batch_size):
            s = Variable(torch.zeros(hidden_size))
            h0 = Variable(torch.zeros(1))
            o = Variable(torch.zeros(output_size))
            for k in range(sequence_length):
                x, y = generate(A, b)
                o, s, p, h0 = model(x, s, o, h0)
                loss = loss + criterion(o, y)
                p_sum = p_sum + p
        p_sum = p_sum/batch_size
        loss = loss/batch_size
        if rank == 0:
            pbar.set_postfix(Error=loss.data[0], Ponder=p_sum.data[0])
            losses.append(loss.data[0])
            ponders.append(p_sum.data[0])
        loss = loss + tau*p_sum
        loss.backward()
        optimizer.step()
    if rank == 0:
        q.put(np.array(losses))
        q.put(np.array(ponders))


def exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pd.stats.moments.ewma(values, span=smoothing_factor)


model = ARNN(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
A = 1*torch.randn(output_size, input_size)
b = 1*torch.randn(output_size)
print('=> Launching {} processes'.format(num_processes))
q = mp.SimpleQueue()
model.share_memory()
processes = []
for rank in range(num_processes):
    p = mp.Process(target=train_loop,
                   args=(rank, model, criterion, optimizer, tau, q))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

losses = q.get()
ponders = q.get()

x = np.arange(losses.shape[0])

plt.subplot(2, 1, 1)
plt.plot(x, losses, 'g')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(x, ponders, 'b')
plt.xlabel('Steps')
plt.ylabel('Ponder')

plt.show()

