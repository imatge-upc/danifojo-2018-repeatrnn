import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot script')
parser.add_argument('--dir', type=str, default='./results/parity.pth.tar', metavar='PATH',
                    help='path to the checkpoint (default: ./results/parity.pth.tar)')
parser.add_argument('--span', type=int, default=51, metavar='N',
                    help='span for the exponential moving average (default: 50)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many steps between each checkpoint (default: 10)')


def exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pd.stats.moments.ewma(values, span=smoothing_factor)


args = parser.parse_args()

checkpoint = torch.load(args.dir)
accuracies = exponential_moving_average_smoothing(np.array(checkpoint['accuracies']), args.span)
losses = exponential_moving_average_smoothing(np.array(checkpoint['losses']), args.span)
ponders = exponential_moving_average_smoothing(np.array(checkpoint['ponders']), args.span)

x = np.arange(accuracies.shape[0])*args.log_interval

plt.subplot(3, 1, 1)
plt.plot(x, accuracies, 'r')
plt.ylabel('Accuracy')

plt.subplot(3, 1, 2)
plt.plot(x, losses, 'g')
plt.ylabel('Loss')

plt.subplot(3, 1, 3)
plt.plot(x, ponders, 'b')
plt.xlabel('Steps')
plt.ylabel('Ponder')

plt.show()
