from __future__ import print_function, division
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.multiprocessing as mp
from tqdm import trange

from ACT import ARNN as ARNN

# Training settings
parser = argparse.ArgumentParser(description='Parity task')
parser.add_argument('--input-size', type=int, default=64, metavar='N',
                    help='input size for training (default: 64)')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                    help='hidden size for training (default: 128)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--steps', type=int, default=1000000, metavar='N',
                    help='number of args.steps to train (default: 1000000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many steps between each checkpoint (default: 10)')
parser.add_argument('--start-step', default=0, type=int, metavar='N',
                    help='manual step number (useful on restarts) (default: 0)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--num-processes', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 16)')
parser.add_argument('--tau', type=float, default=1e-3, metavar='TAU',
                    help='value of the time penalty tau (default: 0.001)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=False, type=bool, metavar='True',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

output_size = 1
sequence_length = 1
num_layers = 1


def generate():
    x = np.random.randint(3, size=(args.batch_size, args.input_size)) - 1
    y = np.zeros((args.batch_size,))
    for i in range(args.batch_size):
        unique, counts = np.unique(x[i, :], return_counts=True)
        try:
            y[i] = dict(zip(unique, counts))[1] % 2
        except:
            y[i] = 0
    x = Variable(torch.from_numpy(x).float(), requires_grad=False)
    y = Variable(torch.from_numpy(y).float(), requires_grad=False)
    if args.gpu and torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    return x, y


def accuracy(out, y):
    out = out.view(-1)
    return 1 - torch.sum(torch.abs(y-out))/args.batch_size


def train_loop(rank, args, model, criterion, optimizer, losses, accuracies, ponders):
    np.random.seed()

    if rank == 0:
        loop = trange(args.start_step, args.steps, total=args.steps, initial=args.start_step)
    else:
        loop = range(args.start_step, args.steps)
    for i in loop:
        model.zero_grad()
        outputs = []
        pond_sum = Variable(torch.zeros(1))
        x, y = generate()
        for j in range(args.batch_size):
            s = Variable(torch.zeros(args.hidden_size))
            out, s, p = model(x[j], s)
            outputs.append(out)
            pond_sum = pond_sum + p
        pond_sum = pond_sum/args.batch_size
        out_tensor = torch.cat(outputs)
        loss = criterion(out_tensor, y) + args.tau*pond_sum
        loss.backward()
        optimizer.step()
        if rank == 0 and i % args.log_interval == 0:
            acc = accuracy(torch.round(torch.sigmoid(out_tensor.data)), y.data)
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
            torch.save(checkpoint, './results/parity.pth.tar')


def main():
    print('=> {} cores available'.format(mp.cpu_count()))
    if mp.cpu_count() < args.num_processes:
        args.num_processes = mp.cpu_count()
    model = ARNN(args.input_size, args.hidden_size, output_size)

    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
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


if __name__ == "__main__":
    main()
