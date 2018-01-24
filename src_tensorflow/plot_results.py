import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

tau = ['1.0', '0.5', '0.1', '0.05', '0.01', '0.005', '0.001']
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Parity ACT:
fig, [ax, bx] = plt.subplots(2, 1, sharex='col', dpi=160)
ax.set_title('Accuracy')
bx.set_title('Ponder')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 0.8, len(tau)))
labels = list()
for t, color in zip(tau, colors):
    data = (np.loadtxt('results/parity/act/acc/run_parity_LR=0.001_Len=64_Tau={},tag_Accuracy.csv'.format(t),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    a, = ax.plot(x, y, color=color)
    labels.append(a)
    data = (np.loadtxt('results/parity/act/ponder/run_parity_LR=0.001_Len=64_Tau={},tag_Ponder.csv'.format(t),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    bx.plot(x, y, color=color)
fig.legend(labels ,tau, title=r'Time penalty \tau', ncol=1, fancybox=True, shadow=True)
plt.show()
fig.savefig('plots/parity-act.png')

# Addition ACT:
fig, [ax, bx] = plt.subplots(2, 1, sharex='col')
ax.set_title('Accuracy')
bx.set_title('Ponder')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 0.8, len(tau)))
labels = list()
for t, color in zip(tau, colors):
    data = (np.loadtxt('results/addition/act/acc/run_parity_LR=0.001_Len=64_Tau={},tag_Accuracy.csv'.format(t),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    a, = ax.plot(x, y, color=color)
    labels.append(a)
    data = (np.loadtxt('results/addition/act/ponder/run_parity_LR=0.001_Len=64_Tau={},tag_Ponder.csv'.format(t),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    bx.plot(x, y, color=color)
fig.legend(labels ,tau, title=r'Time penalty \tau', ncol=1, fancybox=True, shadow=True)
plt.show()
fig.savefig('plots/addition-act.png')