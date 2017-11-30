from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import pandas
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

from ACT2 import ARNN

input_size = 64
batch_size = 128
hidden_size = 128
output_size = 1
sequence_length = 1
num_layers = 1
steps = 1000
learning_rate = 0.001
lamda = 1e-5
num_processes = 16


def generate():
    x = np.random.randint(3, size=(batch_size, input_size)) - 1
    y = np.zeros((batch_size,))
    for i in range(batch_size):
        unique, counts = np.unique(x[i, :], return_counts=True)
        try:
            y[i] = dict(zip(unique, counts))[1] % 2
        except:
            y[i] = 0
    x = Variable(torch.from_numpy(x).float())
    y = Variable(torch.from_numpy(y).float())
    return x, y


def _exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pandas.stats.moments.ewma(values, span=smoothing_factor)


def accuracy(out, y):
    out = out.view(-1)
    return 1 - torch.sum(torch.abs(y-out))/batch_size


def train_loop(step, model, criterion, optimizer):
    np.random.seed()
    model.zero_grad()
    outputs = []
    pond_sum = Variable(torch.zeros(1))
    x, y = generate()
    for j in range(batch_size):
        s = Variable(torch.zeros(hidden_size))
        out, s, p = model(x[j], s)
        outputs.append(torch.round(F.sigmoid(out)))
        # print(p)
        pond_sum = pond_sum + p
    out_tensor = torch.cat(outputs)
    loss = criterion(out_tensor, y) + lamda*pond_sum
    loss.backward()
    optimizer.step()
    acc = accuracy(out_tensor.data, y.data)
    print('Step {}, loss = {}, accuracy = {}'.format(step+1, loss.data[0], acc))


def main():
    model = ARNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = np.zeros(steps)
    acc = np.zeros(steps)
    ponders = np.zeros(steps)
    model.share_memory()
    for i in range(steps):
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train_loop, args=(i, model, criterion, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
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


if __name__ == "__main__":
    main()
