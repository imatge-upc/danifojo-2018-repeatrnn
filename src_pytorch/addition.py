from __future__ import print_function, division
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.multiprocessing as mp
from tqdm import trange

from act_cell import ALSTM as ALSTM

# Training settings
parser = argparse.ArgumentParser(description='Addition task')
parser.add_argument('--sequence-length', type=int, default=5, metavar='N',
                    help='sequence length for training (default: 5)')
parser.add_argument('--total-digits', type=int, default=5, metavar='N',
                    help='total digits for training (default: 5)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
                    help='hidden size for training (default: 512)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--steps', type=int, default=2000000, metavar='N',
                    help='number of args.steps to train (default: 2000000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many steps between each checkpoint (default: 10)')
parser.add_argument('--start-step', default=0, type=int, metavar='N',
                    help='manual step number (useful on restarts) (default: 0)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--num-processes', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 16)')
parser.add_argument('--tau', type=float, default=1e-3, metavar='TAU',
                    help='value of the time penalty tau (default: 0.001)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

input_size = 10*args.total_digits
output_size = 11*(args.total_digits+1)
num_layers = 1

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
    input_dec = np.zeros((args.batch_size, args.sequence_length, args.total_digits), dtype=int) + 10
    input_enc = np.zeros((args.batch_size, args.sequence_length, input_size), dtype=int)
    output_dec = np.zeros((args.batch_size, args.sequence_length, args.total_digits+1), dtype=int)
    output_enc = np.zeros((args.batch_size, args.sequence_length, output_size), dtype=int)
    for i in range(args.batch_size):
        for j in range(args.sequence_length):
            digits = np.random.randint(args.total_digits) + 1
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
    x = Variable(torch.from_numpy(input_enc), requires_grad=False).float()
    y = Variable(torch.from_numpy(output_dec), requires_grad=False).long()
    return x, y


def accuracy(out, y):
    _, indices = torch.max(out, 1)
    s = 0
    for i in range(args.batch_size*(args.sequence_length-1)):
        if torch.eq(indices[i*(args.total_digits+1): (i+1)*(args.total_digits+1)],
                    y[i*(args.total_digits+1): (i+1)*(args.total_digits+1)]).all():
            s += 1
    return s/(args.batch_size*(args.sequence_length-1))


def train_loop(rank, args, model, criterion, optimizer, losses, accuracies, ponders):
    np.random.seed()

    if rank == 0:
        loop = trange(args.start_step, args.steps//args.num_processes, total=args.steps//args.num_processes, initial=args.start_step)
    else:
        loop = range(args.start_step, args.steps//args.num_processes)
    for i in loop:
        model.zero_grad()
        outputs = []
        pond_sum = Variable(torch.zeros(1))
        x, y = generate()
        for j in range(args.batch_size):
            s = (Variable(torch.zeros(args.hidden_size)), Variable(torch.zeros(args.hidden_size)))
            for k in range(args.sequence_length):
                out, s, p = model(x[j, k], s)
                outputs.append(out)
                pond_sum = pond_sum + p
        pond_sum = pond_sum/args.batch_size
        out_tensor = torch.stack(outputs, dim=0)
        out_tensor = out_tensor.view(args.batch_size, args.sequence_length, output_size)
        out_tensor_masked = out_tensor[:, 1:, :].contiguous().view(-1, 11)
        y_masked = y[:, 1:].contiguous().view(-1)
        loss = criterion(out_tensor_masked, y_masked) + args.tau*pond_sum
        loss.backward()
        optimizer.step()
        if rank == 0 and i % args.log_interval == 0:
            acc = accuracy(out_tensor_masked.data, y_masked.data)
            loop.set_postfix(Loss='{:0.3f}'.format(loss.data[0]),
                             Accuracy='{:0.3f}'.format(acc),
                             Ponder='{:0.3f}'.format(pond_sum.data[0]))
            losses.append(loss.data[0])
            accuracies.append(acc)
            ponders.append(pond_sum.data[0])
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
                'accuracies': accuracies,
                'ponders': ponders,
                'step': i + 1}
            torch.save(checkpoint, './results/addition.pth.tar')


def main():
    print('=> {} cores available'.format(mp.cpu_count()))
    if mp.cpu_count() < args.num_processes:
        args.num_processes = mp.cpu_count()

    model = ALSTM(input_size, args.hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    accuracies = []
    ponders = []

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_step = checkpoint['step']
            losses = checkpoint['losses']
            accuracies = checkpoint['accuracies']
            ponders = checkpoint['ponders']
            print('=> loaded checkpoint {} (step {})'
                  .format(args.resume, checkpoint['step']))
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    print('=> Launching {} processes'.format(args.num_processes))
    model.share_memory()
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train_loop,
                       args=(rank, args, model, criterion, optimizer, losses, accuracies, ponders))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
