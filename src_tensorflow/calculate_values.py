import numpy as np
from scipy.signal import savgol_filter as filt

tau = ['1.0', '0.5', '0.1', '0.05', '0.01', '0.005', '0.001']
pon = range(1, 13)

# PARITY ACT
# Find minima
print('PARITY ACT')
for t in tau:
    x = np.loadtxt('./results/parity/act/acc/run_parity_LR=0.001_Len=64_Tau={},tag_Accuracy.csv'.format(t), delimiter=',', skiprows=1)[:, 2]
    valid=np.where(filt(x, 71, 3)>0.98)[0]
    try:
        print('Tau={}, minimum={}'.format(t, valid[x[valid].argmin()]*1000))
    except:
        print('Tau={}, not solved'.format(t))

# Find mean
print('\n')
for t in tau:
    x = np.loadtxt('./results/parity/act/ponder/run_parity_LR=0.001_Len=64_Tau={},tag_Ponder.csv'.format(t), delimiter=',', skiprows=1)[:, 2]
    print('Tau={}, mean={}'.format(t, np.mean(np.floor(x))))

print('\n')
print('PARITY REPEAT')
for p in pon:
    x = np.loadtxt('./results/parity/repeat/run_parity_test_LR=0.001_Len=64_Pond={},tag_Accuracy.csv'.format(p), delimiter=',', skiprows=1)[:, 2]
    valid=np.where(filt(x, 71, 3)>0.98)[0]
    try:
        print('Ponder={}, minimum={}'.format(p, valid[x[valid].argmin()]*1000))
    except:
        print('Ponder={}, not solved'.format(p))


# ADDITION ACT
# Find minima
print('\n')
print('\n')
print('ADDITION ACT')

for t in tau:
    x = np.loadtxt('./results/addition/act/acc/run_addition_LR=0.001_Tau={},tag_Accuracy.csv'.format(t), delimiter=',', skiprows=1)[:, 2]
    valid=np.where(filt(x, 71, 3)>0.98)[0]
    try:
        print('Tau={}, minimum={}'.format(t, valid[x[valid].argmin()]*1000))
    except:
        print('Tau={}, not solved'.format(t))
# Find mean
print('\n')

for t in tau:
    x = np.loadtxt('./results/addition/act/ponder/run_addition_LR=0.001_Tau={},tag_Ponder.csv'.format(t), delimiter=',', skiprows=1)[:, 2]
    print('Tau={}, mean={}'.format(t, np.mean(np.floor(x))))

print('\n')
print('ADDITION REPEAT')
for p in pon:
    x = np.loadtxt('./results/addition/repeat/run_addition_test_LR=0.001_Pond={},tag_Accuracy.csv'.format(p),
                   delimiter=',', skiprows=1)[:, 2]
    valid = np.where(filt(x, 71, 3) > 0.98)[0]
    try:
        print('Ponder={}, minimum={}'.format(p, valid[x[valid].argmin()] * 1000))
    except:
        print('Ponder={}, not solved'.format(p))