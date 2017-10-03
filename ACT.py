from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class ACT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, eps=0.01):
        super(ACT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.eps = eps
        self.ponder = Parameter(torch.zeros(0))
        self.fc_halt = nn.Linear(input_size+hidden_size, 1)
        self.fc_hidden = nn.Linear(input_size+hidden_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        outputs = Variable(torch.Tensor(1, self.output_size))
        states = Variable(torch.Tensor(1, self.hidden_size))
        halt_prob = Variable(torch.Tensor(1))

        # First iteration
        n = 0
        z = torch.cat((x+1, h))
        states[0] = self.fc_hidden(z)
        halt_prob[0] = F.sigmoid(self.fc_halt(states[0]))
        outputs[0] = self.fc_output(states[0])
        halt = halt_prob[0]

        while halt < 1-self.eps:
            n += 1
            z = torch.cat((x, states[n, :]))
            states = torch.cat((states, self.fc_hidden(z)).view(1, -1))
            halt_prob = torch.cat((halt_prob, F.sigmoid(self.fc_halt(states[n]))))
            outputs = torch.cat((outputs, self.fc_output(states[n]).view(1, -1)))
            halt += halt_prob.data[n]

        halt_prob[n] = 1 - torch.sum(halt_prob[:-1])

        output = torch.mm(halt_prob.view(1, -1), outputs).view(-1)
        hidden_state = torch.mm(halt_prob.view(1, -1), states).view(-1)
        ponder[0] = n + 1 + halt_prob[n]

        return output, hidden_state, ponder

    def backward(self, grad_output):
        pass
