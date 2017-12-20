from __future__ import print_function, division
import time
import logging

import numpy as np
import pandas
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from gpu import ACT

# Configure logger
# logging.basicConfig(filename='/results/add10.log', filemode='w', level=logging.DEBUG)

# Hyperparameters
sequence_length = 5
total_digits = 5
input_size = 10*total_digits
output_size = 11*(total_digits+1)
batch = 32
hidden_size = 512
num_layers = 1
num_epochs = 10000
learning_rate = 0.001


# Generate samples
def add_vec(x, y):
    n = max(len(x), len(y))
    return num2vec(vec2num(x) + vec2num(y), n)


def vec2num(x):
    s = 0
    for i in range(len(x)):
        if x[i] == 10:
            break
        s *= 10
        s += x[i]
    return s


def num2vec(x, n):
    y = np.zeros(n) + 10
    digits = len(str(int(x)))
    for i in range(digits):
        y[i] = (x//10**(digits-i-1)) % 10
    return y


def encode_in(x):
    y = np.zeros(len(x)*10)
    for i in range(len(x)):
        if x[i] == 10:
            break
        else:
            y[10*i+int(x[i])] = 1
    return y


def encode_out(x):
    y = np.zeros(len(x)*11, dtype=int)
    for i in range(len(x)):
        if x[i] == 10:
            y[11*i+10] = 1
        else:
            y[11*i+int(x[i])] = 1
    return y


def decode_out(x):
    y = np.zeros(len(x)//11, dtype=int)
    for i in range(len(y)):
        y[i] = np.argmax(x[i*11:(i+1)*11])
    return y


def generate():
    input_dec = np.zeros((batch, sequence_length, total_digits), dtype=int) + 10
    input_enc = np.zeros((batch, sequence_length, input_size), dtype=int)
    output_dec = np.zeros((batch, sequence_length, total_digits+1), dtype=int)
    output_enc = np.zeros((batch, sequence_length, output_size), dtype=int)
    for i in range(batch):
        for j in range(sequence_length):
            digits = np.random.randint(total_digits) + 1
            for k in range(digits):
                d = np.random.randint(10)
                input_dec[i, j, k] = d
            if j == 0:
                output_dec[i, j, :-1] = input_dec[i, j, :]
                output_dec[i, j, -1] = 10
            elif j > 0:
                output_dec[i, j, :] = add_vec(output_dec[i, j-1, :], input_dec[i, j, :])
            input_enc[i, j, :] = encode_in(input_dec[i, j, :])
            output_enc[i, j, :] = encode_out(output_dec[i, j, :])
    x = Variable(torch.from_numpy(input_enc)).float()
    y = Variable(torch.from_numpy(output_dec)).long()
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    return x, y


# Softmax function that can act in an specified axis. Code snippet from:
# https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
def softmax(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


# Declare model
class ALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.arnn = ACT.ALSTM(input_size, hidden_size, output_size, num_layers, batch_first=True)

    def forward(self, x):
        (h0, c0) = self.init_hidden_state()
        out, p = self.arnn(x, (h0, c0))
        out = out.contiguous()
        for i in range(total_digits+1):
            out[:, :, i*11:(i+1)*11] = \
                softmax(out[:, :, i*11:(i+1)*11].clone(), axis=2)
        return out, p

    def init_hidden_state(self):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
        return h0, c0


arnn = ALSTM(input_size, hidden_size, output_size, num_layers)

if torch.cuda.is_available():
    arnn.cuda()


# Loss that takes into account every digit of every element of the sequence.
def CustomLoss(x, y, criterion):
    s = Variable(torch.zeros(1), requires_grad=False)
    if torch.cuda.is_available():
        s = s.cuda()
    y = y.view(batch, -1)
    x = x.view(batch, -1)
    for j in range(1, sequence_length):
        for i in range(total_digits+1):
            s = s + criterion(x[:, output_size*j + i*11:output_size*j +
                                (i+1)*11], y[:, (total_digits+1)*j + i])
    return s


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(arnn.parameters(), lr=learning_rate)


# Function to calculate accuracy.
def accuracy(out, y):
    if torch.cuda.is_available():
        out = out.view(batch, sequence_length, -1).cpu().data.numpy()
        y = y.view(batch, sequence_length, -1).cpu().data.numpy()
    else:
        out = out.view(batch, sequence_length, -1).data.numpy()
        y = y.view(batch, sequence_length, -1).data.numpy()
    s = 0.
    for k in range(batch):
        for j in range(1, sequence_length):
            dec = decode_out(out[k, j, :])
            if np.allclose(dec, y[k, j, :]):
                s += 1
    return s/((sequence_length-1)*batch)


# Smoothing function for plots.
def _exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pandas.stats.moments.ewma(values, span=smoothing_factor)


# Training.
losses = np.array([])
acc = np.array([])
last_time = time.time()
logging.info('Starting training.')
for i in range(num_epochs):
    arnn.zero_grad()
    x, y = generate()
    out, p = arnn(x)
    acc = np.append(acc, accuracy(out, y))
    loss = CustomLoss(out, y, criterion) + torch.sum(p)
    losses = np.append(losses, loss.data[0])
    loss.backward()
    optimizer.step()
    if (i+1) % (num_epochs//100) == 0:
        logging.info('Step ' + str(i + 1) + '/' + str(num_epochs) + '. ' +
                     'Loss = ' + str(losses[i]) + '. ' +
                     'Accuracy = ' + str(acc[i]) + '. ' +
                     'Time elapsed: ' + str(time.time()-last_time) + '.')
        print('Step ' + str(i + 1) + '/' + str(num_epochs) + '. ' +
                     'Loss = ' + str(losses[i]) + '. ' +
                     'Accuracy = ' + str(acc[i]) + '. ' +
                     'Time elapsed: ' + str(time.time()-last_time) + '.')
        last_time = time.time()
        np.save('./add10_loss.npy', losses)
        np.save('./add10_acc.npy', acc)
        torch.save(arnn.state_dict(), './add10_model.torch')
logging.info('Finished training.')

# Plots
try:
    losses_f = _exponential_moving_average_smoothing(losses, 50)
    plt.plot(losses_f)
    plt.title('Loss')
    plt.figure()
    acc_f = _exponential_moving_average_smoothing(acc, 50)
    plt.plot(acc_f)
    plt.title('Accuracy')
except:
    logging.info('No Display available.')
