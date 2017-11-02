from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import matplotlib.pyplot as plt

from ACT import ARNN

batch_size = 10
sequence_length = 4
input_size = 3
hidden_size = 5
output_size = 2
learning_rate = 0.01
steps = 1000
lamda = 1e-3


def generate():
    x = Variable(torch.randn(batch_size, sequence_length, input_size))
    y = Variable(torch.zeros(batch_size, sequence_length, output_size), requires_grad=False)
    A = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    for i in range(batch_size):
        for j in range(sequence_length):
            y.data[i, j, :] = torch.mm(A, x.data[i, j, :].view(input_size, 1))
    return x, y


x, y = generate()
h = Variable(torch.zeros(1, batch_size, hidden_size))
arnn = ARNN(input_size, hidden_size, output_size, batch_first=True)
o, p = arnn(x, h)
print(o)
print(p)

optimizer = torch.optim.Adam(arnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
losses = np.zeros(steps)

for i in range(steps):
    arnn.zero_grad()
    x, y = generate()
    o, p = arnn(x, h)
    loss = criterion(o, y) + lamda*torch.sum(p)
    losses[i] = loss.data[0]
    loss.backward()
    optimizer.step()

x, y = generate()
o, p = arnn(x, h)
print(o)
print(y)
plt.plot(losses)
plt.show()
