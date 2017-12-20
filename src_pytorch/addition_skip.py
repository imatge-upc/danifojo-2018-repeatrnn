from __future__ import print_function, division
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.multiprocessing as mp
from tqdm import trange

from ACT import ALSTM as ALSTM

# Training settings
parser = argparse.ArgumentParser(description='Parity task')
parser.add_argument('--sequence-length', type=int, default=50, metavar='N',
                    help='sequence length for training (default: 50)')
parser.add_argument('--hidden-size', type=int, default=110, metavar='N',
                    help='hidden size for training (default: 110)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
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

args = parser.parse_args()

input_size = 2
output_size = 1
num_layers = 1


def generate():
    x = np.stack((np.random.rand(args.batch_size, args.sequence_length) - 0.5,
                  np.zeros((args.batch_size, args.sequence_length))), axis=2)
    y = np.zeros(args.batch_size)
    n1 = np.random.randint(0, args.sequence_length//10+1, (args.batch_size,))
    n2 = np.random.randint(args.sequence_length//2, args.sequence_length, (args.batch_size,))
    for i in range(args.batch_size):
        x[i, n1[i], 1] = 1
        x[i, n2[i], 1] = 1
        y[i] = x[i, n1[i], 0] + x[i, n2[i], 0]
    return Variable(torch.from_numpy(x), requires_grad=False).float(), Variable(torch.from_numpy(y), requires_grad=False).float()


def train_loop(rank, args, model, criterion, optimizer, losses, ponders):
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
            s = (Variable(torch.zeros(args.hidden_size)), Variable(torch.zeros(args.hidden_size)))
            for k in range(args.sequence_length):
                out, s, p = model(x[j, k], s)
                pond_sum = pond_sum + p
            outputs.append(out)
        pond_sum = pond_sum/args.batch_size
        out_tensor = torch.cat(outputs)
        loss = criterion(out_tensor, y) + args.tau*pond_sum
        loss.backward()
        optimizer.step()
        if rank == 0 and i % args.log_interval == 0:
            loop.set_postfix(Loss='{:0.3f}'.format(loss.data[0]),
                             Ponder='{:0.3f}'.format(pond_sum.data[0]))
            losses.append(loss.data[0])
            ponders.append(pond_sum.data[0])
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
                'ponders': ponders,
                'step': i + 1}
            torch.save(checkpoint, './results/addition_skip.pth.tar')


def main():
    print('=> {} cores available'.format(mp.cpu_count()))
    if mp.cpu_count() < args.num_processes:
        args.num_processes = mp.cpu_count()

    model = ALSTM(input_size, args.hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    ponders = []

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_step = checkpoint['step']
            losses = checkpoint['losses']
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
                       args=(rank, args, model, criterion, optimizer, losses, ponders))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    x, y = generate()
    main()
