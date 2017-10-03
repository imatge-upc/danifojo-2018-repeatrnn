import torch
import matplotlib.pyplot as plt
import numpy as np
a = torch.Tensor(1, 3)

x = np.linspace(-np.pi, np.pi, 100)

y = np.cosh(x)

plt.plot(x, y)
