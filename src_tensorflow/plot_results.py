import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# import seaborn as sns
# sns.set_style("whitegrid")

tau = ['1.0', '0.5', '0.1', '0.05', '0.01', '0.005', '0.001']
repeats = range(1, 13)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Parity ACT:
fig, [ax, bx] = plt.subplots(2, 1, sharex='col', figsize=(12, 6), dpi=80)
ax.set_title('Accuracy')
ax.grid(linestyle='--')
bx.set_title('Ponder')
bx.grid(linestyle='--')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0.2, 0.8, len(tau)))
colors2 = cmap(np.linspace(0.2, 0.8, len(repeats)))
labels = list()

data = (np.loadtxt('results/parity/run_parity_LR=0.001_Len=64_NoACT,tag_Accuracy.csv',
                   delimiter=',', skiprows=1))
x = data[:, 1]
y = data[:, 2]
y = savgol_filter(y, 71, 3)
a, = ax.plot(x, y, color='k')
labels.append(a)

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
fig.legend(labels, ['Without ACT'] + tau, title=r'Time penalty \tau', ncol=1, fancybox=False, shadow=False)
# plt.show()
fig.savefig('plots/parity-act.pdf')
fig.savefig('plots/parity-act.png')

# Parity repeat:
fig, ax = plt.subplots(1, 1, sharex='col', figsize=(12, 6), dpi=80)
ax.set_title('Accuracy')
ax.grid(linestyle='--')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0.2, 0.8, len(tau)))
labels = list()

for p, color in zip(repeats, colors2):
    data = (np.loadtxt('results/parity/repeat/run_parity_test_LR=0.001_Len=64_Pond={},tag_Accuracy.csv'.format(p),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    a, = ax.plot(x, y, color=color)
    labels.append(a)
fig.legend(labels, repeats, title=r'Amount of repetitions \rho', ncol=1, fancybox=False, shadow=False)
# plt.show()
fig.savefig('plots/parity-repeat.pdf')
fig.savefig('plots/parity-repeat.png')

# Addition ACT:
fig, [ax, bx] = plt.subplots(2, 1, sharex='col', figsize=(12, 6), dpi=80)
ax.set_title('Accuracy')
ax.grid(linestyle='--')
bx.set_title('Ponder')
bx.grid(linestyle='--')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0.2, 0.8, len(tau)))
labels = list()

data = (np.loadtxt('results/addition/repeat/run_addition_test_LR=0.001_Pond=1,tag_Accuracy.csv',
                   delimiter=',', skiprows=1))
x = data[:, 1]
y = data[:, 2]
y = savgol_filter(y, 71, 3)
a, = ax.plot(x, y, color='k')
labels.append(a)

for t, color in zip(tau, colors):
    data = (np.loadtxt('results/addition/act/acc/run_addition_LR=0.001_Tau={},tag_Accuracy.csv'.format(t),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    a, = ax.plot(x, y, color=color)
    labels.append(a)
    data = (np.loadtxt('results/addition/act/ponder/run_addition_LR=0.001_Tau={},tag_Ponder.csv'.format(t),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    bx.plot(x, y, color=color)
fig.legend(labels, ['Without ACT'] + tau, title=r'Time penalty \tau', ncol=1, fancybox=False, shadow=False)
# plt.show()
fig.savefig('plots/addition-act.pdf')
fig.savefig('plots/addition-act.png')

# Addition repeat:
fig, ax = plt.subplots(1, 1, sharex='col', figsize=(12, 6), dpi=80)
ax.set_title('Accuracy')
ax.grid(linestyle='--')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0.2, 0.8, len(tau)))
labels = list()

for p, color in zip(repeats, colors2):
    data = (np.loadtxt('results/addition/repeat/run_addition_test_LR=0.001_Pond={},tag_Accuracy.csv'.format(p),
                                     delimiter=',', skiprows=1))
    x = data[:, 1]
    y = data[:, 2]
    y = savgol_filter(y, 71, 3)
    a, = ax.plot(x, y, color=color)
    labels.append(a)
fig.legend(labels, repeats, title=r'Amount of repetitions \rho', ncol=1, fancybox=False, shadow=False)
# plt.show()
fig.savefig('plots/addition-repeat.pdf')
fig.savefig('plots/addition-repeat.png')

# Addition ponder:
fig, ax = plt.subplots(1, 1, sharex='col', figsize=(12, 6), dpi=80)
ax.set_title('Ponder')
ax.grid(linestyle='--')

data = (np.loadtxt('ponders_addition.txt', delimiter=' '))
x = data[-50:]
ax.boxplot(x)
# plt.show()
fig.savefig('plots/addition-ponders.pdf')
fig.savefig('plots/addition-ponders.png')
