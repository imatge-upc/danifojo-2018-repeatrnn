from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class Binarize(torch.autograd.Function):
    def forward(self, input):
        input[input > 0] = 1
        return input

    def backward(self, grad_output):
        return grad_output


class Simple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, eps=0.01):
        super(Simple, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.eps = eps
        self.ponder = Variable(torch.zeros(1))
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_hidden = nn.Linear(1+input_size+hidden_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, s):
        outputs = Variable(torch.Tensor(1, self.output_size))
        states = Variable(torch.Tensor(1, self.hidden_size))
        halt_prob = Variable(torch.Tensor(1))

        # First iteration
        n = 0
        x0 = torch.cat((Variable(torch.Tensor([0])), x))
        x1 = torch.cat((Variable(torch.Tensor([1])), x))
        z = torch.cat((x1, s))
        states[0] = self.fc_hidden(z)
        halt_prob[0] = F.sigmoid(self.fc_halt(states[0]))
        outputs[0] = self.fc_output(states[0])
        halt = halt_prob.data[0]

        while halt < 1-self.eps:
            n += 1
            z = torch.cat((x0, states[n-1, :]))
            states = torch.cat((states, self.fc_hidden(z).view(1, -1)))
            halt_prob = torch.cat((halt_prob, F.sigmoid(self.fc_halt(states[n]))))
            outputs = torch.cat((outputs, self.fc_output(states[n]).view(1, -1)))
            halt += halt_prob.data[n]

        halt_prob[n] = 1 - torch.sum(halt_prob[:-1])

        output = torch.mm(halt_prob.view(1, -1), outputs).view(-1)
        hidden_state = torch.mm(halt_prob.view(1, -1), states).view(-1)
        self.ponder[0] = n + 1 + halt_prob[n]
        return output, hidden_state, self.ponder


class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, eps=0.01, batch_first=False):
        super(ARNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.eps = eps
        self.batch_first = batch_first
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.rnn = nn.RNN(1+input_size, hidden_size, num_layers=self.num_layers,
                          batch_first=self.batch_first)

    def forward(self, x, s):
        if not self.batch_first:
            x = torch.transpose(x, 0, 1)

        batch_size = x.size()[0]
        sequence_length = x.size()[1]
        output = Variable(torch.Tensor(batch_size, sequence_length, self.output_size))
        hidden_state = Variable(torch.Tensor(batch_size, sequence_length, self.hidden_size))
        ponder = Variable(torch.Tensor(batch_size, sequence_length))
        x0 = torch.cat((Variable(torch.zeros(x.size()[0], x.size()[1], 1)), x), 2)
        x1 = torch.cat((Variable(torch.zeros(x.size()[0], x.size()[1], 1))+1, x), 2)

        for i in range(sequence_length):
            # First iteration
            outputs = []
            states = []
            halt_prob = []
            pond = Variable(torch.zeros(batch_size))
            n = 0
            s = self.rnn(x1[:, i, :].contiguous().view(batch_size, 1, -1), s)[0]
            out = self.fc_output(s)
            states.append(s)
            outputs.append(out)
            p = F.sigmoid(self.fc_halt(s))
            halt_prob.append(p)
            halt = halt_prob[0].data.numpy().reshape(-1)
            for j in range(batch_size):
                if halt[j] >= 1-self.eps:
                    pond[j] = 1 + (n+1)
                    halt_prob[n][j, 0, 0] = 1

            while np.any(halt < 1-self.eps):
                n += 1
                s = self.rnn(x0[:, i, :].contiguous().view(batch_size, 1, -1), states[n-1])[0]
                out = self.fc_output(s)
                p = F.sigmoid(self.fc_halt(s))
                for j in range(batch_size):
                    if halt[j] >= 1-self.eps:
                        out[j, :, :] = out[j, :, :]*0
                        s[j, :, :] = s[j, :, :]*0
                        p[j, :, :] = p[j, :, :]*0
                halt += p.data.numpy().reshape(-1)
                states.append(s)
                outputs.append(out)
                halt_prob.append(p)
                for j in range(batch_size):
                    if halt[j] >= 1-self.eps and p.data.numpy()[j, :, :] != 0:
                        r = 1 - sum([it.data[j, 0, 0] for it in halt_prob[:-1]])
                        pond[j] = r + (n+1)
                        halt_prob[n][j, 0, 0] = r

            outputs_tensor = torch.stack(outputs, 3)
            states_tensor = torch.stack(states, 3)
            halt_prob_tensor = torch.stack(halt_prob, 3)

            o = torch.bmm(outputs_tensor.view(batch_size, self.output_size, n+1),
                          torch.transpose(halt_prob_tensor, 2, 3).view(batch_size, n+1, 1))
            o = o.view(batch_size, 1, self.output_size)
            output[:, i, :] = o.view(batch_size, self.output_size)

            s = torch.bmm(states_tensor.view(batch_size, self.hidden_size, n+1),
                          torch.transpose(halt_prob_tensor, 2, 3).view(batch_size, n+1, 1))
            s = s.view(1, batch_size, self.hidden_size)

            ponder[:, i] = pond

        if not self.batch_first:
            output = output.transpose(0, 1)
            ponder = ponder.transpose(0, 1)
        return output, ponder


class ARNN_bin(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, eps=0.01, batch_first=False):
        super(ARNN_bin, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.eps = eps
        self.batch_first = batch_first
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.rnn = nn.RNN(1+input_size, hidden_size, num_layers=self.num_layers,
                          batch_first=self.batch_first)
        self.f_bin = Binarize()

    def forward(self, x, s):
        if not self.batch_first:
            x = torch.transpose(x, 0, 1)

        batch_size = x.size()[0]
        sequence_length = x.size()[1]
        output = Variable(torch.Tensor(batch_size, sequence_length, self.output_size))
        hidden_state = Variable(torch.Tensor(batch_size, sequence_length, self.hidden_size))
        ponder = Variable(torch.Tensor(batch_size, sequence_length))
        x0 = torch.cat((Variable(torch.zeros(x.size()[0], x.size()[1], 1)), x), 2)
        x1 = torch.cat((Variable(torch.zeros(x.size()[0], x.size()[1], 1))+1, x), 2)

        for i in range(sequence_length):
            # First iteration
            outputs = []
            states = []
            halt_prob = []
            pond = Variable(torch.zeros(batch_size))
            n = 0
            s = self.rnn(x1[:, i, :].contiguous().view(batch_size, 1, -1), s)[0]
            out = self.fc_output(s)
            states.append(s)
            outputs.append(out)
            p = F.sigmoid(self.fc_halt(s))
            halt_prob.append(p)
            halt = halt_prob[0].data.numpy().reshape(-1)
            for j in range(batch_size):
                if halt[j] >= 1-self.eps:
                    pond[j] = 1 + (n+1)
                    halt_prob[n][j, 0, 0] = 1

            while np.any(halt < 1-self.eps):
                n += 1
                s = self.rnn(x0[:, i, :].contiguous().view(batch_size, 1, -1), states[n-1])[0]
                out = self.fc_output(s)
                p = F.sigmoid(self.fc_halt(s))
                for j in range(batch_size):
                    if halt[j] >= 1-self.eps:
                        out[j, :, :] = out[j, :, :]*0
                        s[j, :, :] = s[j, :, :]*0
                        p[j, :, :] = p[j, :, :]*0
                halt += p.data.numpy().reshape(-1)
                states.append(s)
                outputs.append(out)
                halt_prob.append(p)

            outputs_tensor = torch.stack(outputs, 3)
            states_tensor = torch.stack(states, 3)
            halt_prob_tensor = torch.stack(halt_prob, 3)

            o = torch.bmm(outputs_tensor.view(batch_size, self.output_size, n+1),
                          self.f_bin(torch.transpose(halt_prob_tensor, 2, 3).view(batch_size, n+1, 1)))

            o = o.view(batch_size, 1, self.output_size)
            output[:, i, :] = o.view(batch_size, self.output_size)

            s = torch.bmm(states_tensor.view(batch_size, self.hidden_size, n+1),
                          self.f_bin(torch.transpose(halt_prob_tensor, 2, 3).view(batch_size, n+1, 1)))
            s = s.view(1, batch_size, self.hidden_size)

            ponder[:, i] = pond

        if not self.batch_first:
            output = output.transpose(0, 1)
            ponder = ponder.transpose(0, 1)
        return output, ponder
