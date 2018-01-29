from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ============================================== ACT ===================================================================

class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, eps=0.01, M=100):
        super(ARNN, self).__init__()
        self.eps = eps
        self.M = M
        self.rnn = nn.RNN(1+input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)  # Add one to the input size for the binary flag
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([1.]))  # Set initial bias to avoid to many iterations at the beginning
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, s):
        outputs = []
        states = []
        halt_prob = []
        if isinstance(x.data[0], int):
            x0 = torch.cat((Variable(torch.Tensor([0])), x))
            x1 = torch.cat((Variable(torch.Tensor([1])), x))
        if isinstance(x.data[0], float):
            x0 = torch.cat((Variable(torch.Tensor([0])).float(), x))
            x1 = torch.cat((Variable(torch.Tensor([1])).float(), x))

        # First iteration
        n = 0
        states.append(self.rnn(x1.view(1, 1, -1), s.view(1, 1, -1))[0].view(-1))
        halt_prob.append(F.sigmoid(self.fc_halt(states[0])))
        outputs.append(self.fc_output(states[0]))
        halt_sum = halt_prob[0].data[0]

        while halt_sum < 1-self.eps and n < self.M:
            n += 1
            states.append(self.rnn(x0.view(1, 1, -1), states[n-1].view(1, 1, -1))[0].view(-1))
            halt_prob.append(F.sigmoid(self.fc_halt(states[n])))
            outputs.append(self.fc_output(states[n]))
            halt_sum += halt_prob[n].data[0]

        if len(halt_prob) > 1:
            r = 1 - torch.sum(torch.cat(halt_prob[:-1]))  # Residual
        else:
            r = Variable(torch.Tensor([1]))

        halt_prob[n] = r

        outputs_tensor = torch.stack(outputs, dim=1)
        states_tensor = torch.stack(states, dim=1)
        halt_prob_tensor = torch.cat(halt_prob)

        output = torch.mv(outputs_tensor, halt_prob_tensor)
        s = torch.mv(states_tensor, halt_prob_tensor).view(-1)
        ponder = n + 1 + r
        return output, s, ponder


class ALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, eps=0.01, M=100):
        super(ALSTM, self).__init__()
        self.eps = eps
        self.M = M
        self.lstm = nn.LSTM(1+input_size, hidden_size,
                            num_layers=num_layers, batch_first=True)  # Add one to the input size for the binary flag
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([1.]))  # Set initial bias to avoid problems at the beginning
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, s):
        outputs = []
        states = []
        cells = []
        halt_prob = []
        (h, c) = s
        if isinstance(x.data[0], int):
            x0 = torch.cat((Variable(torch.Tensor([0])), x))
            x1 = torch.cat((Variable(torch.Tensor([1])), x))
        if isinstance(x.data[0], float):
            x0 = torch.cat((Variable(torch.Tensor([0])).float(), x))
            x1 = torch.cat((Variable(torch.Tensor([1])).float(), x))

        # First iteration
        n = 0
        (h0, c0) = self.lstm(x1.view(1, 1, -1), (h.view(1, 1, -1), c.view(1, 1, -1)))[1]
        states.append(h0.view(-1))
        cells.append(c0.view(-1))
        halt_prob.append(F.sigmoid(self.fc_halt(states[0])))
        outputs.append(self.fc_output(states[0]))
        halt_sum = halt_prob[0].data[0]
        while halt_sum < 1-self.eps and n < self.M:
            n += 1
            (h0, c0) = self.lstm(x0.view(1, 1, -1), (states[n-1].view(1, 1, -1), cells[n-1].view(1, 1, -1)))[1]
            states.append(h0.view(-1))
            cells.append(c0.view(-1))
            halt_prob.append(F.sigmoid(self.fc_halt(states[n])))
            outputs.append(self.fc_output(states[n]))
            halt_sum += halt_prob[n].data[0]

        if len(halt_prob) > 1:
            r = 1 - torch.sum(torch.cat(halt_prob[:-1]))  # Residual
        else:
            r = Variable(torch.Tensor([1]))

        halt_prob[n] = r

        outputs_tensor = torch.stack(outputs, dim=1)
        states_tensor = torch.stack(states, dim=1)
        cells_tensor = torch.stack(cells, dim=1)
        halt_prob_tensor = torch.cat(halt_prob)
        output = torch.mv(outputs_tensor, halt_prob_tensor)
        h = torch.mv(states_tensor, halt_prob_tensor).view(-1)
        c = torch.mv(cells_tensor, halt_prob_tensor).view(-1)
        ponder = n + 1 + r
        return output, (h, c), ponder


# ================================================ ACT_B ===============================================================

# This part is for future research. It might not work.

class Binarize(torch.autograd.Function):
    def __init__(self, eps=0.01):
        self.eps = eps

    def forward(self, input):
        input[input >= 1 - self.eps] = 1
        input[input < 1 - self.eps] = 0
        return input

    def backward(self, grad_output):
        return grad_output


class ARNN_B(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, eps=0.01):
        super(ARNN_B, self).__init__()
        self.eps = eps
        self.rnn = nn.RNN(1+input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)  # Add one to the input size for the binary flag
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([1.]))  # Set initial bias to avoid problems at the beginning
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.f_bin = Binarize(eps=eps)

    def forward(self, x, s):
        outputs = []
        states = []
        halt_prob_acc = []
        if isinstance(x.data[0], int):
            x0 = torch.cat((Variable(torch.Tensor([0])), x))
            x1 = torch.cat((Variable(torch.Tensor([1])), x))
        if isinstance(x.data[0], float):
            x0 = torch.cat((Variable(torch.Tensor([0])).float(), x))
            x1 = torch.cat((Variable(torch.Tensor([1])).float(), x))

        # First iteration
        n = 0
        states.append(self.rnn(x1.view(1, 1, -1), s.view(1, 1, -1))[0].view(-1))
        outputs.append(self.fc_output(states[0]))
        halt_prob_acc.append(F.sigmoid(self.fc_halt(states[0])))
        halt_sum = halt_prob_acc[0].data[0]

        while halt_sum < 1-self.eps:
            n += 1
            states.append(self.rnn(x0.view(1, 1, -1), states[n-1].view(1, 1, -1))[0].view(-1))
            outputs.append(self.fc_output(states[n]))
            halt_prob_acc.append(halt_prob_acc[n-1].clone() + F.sigmoid(self.fc_halt(states[n])))
            halt_sum = halt_prob_acc[n].data[0]

        outputs_tensor = torch.stack(outputs, dim=1)
        states_tensor = torch.stack(states, dim=1)
        halt_prob_acc_tensor = torch.cat(halt_prob_acc)
        binarized_halt_tensor = self.f_bin(halt_prob_acc_tensor)

        output = torch.mv(outputs_tensor, binarized_halt_tensor)
        s = torch.mv(states_tensor, binarized_halt_tensor).view(-1)
        ponder = torch.sum(1-binarized_halt_tensor)
        return output, s, ponder


class ALSTM_B(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, eps=0.01):
        super(ALSTM_B, self).__init__()
        self.eps = eps
        self.lstm = nn.LSTM(1+input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)  # Add one to the input size for the binary flag
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([1.]))  # Set initial bias to avoid problems at the beginning
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.f_bin = Binarize(eps=eps)

    def forward(self, x, s):
        outputs = []
        states = []
        cells = []
        halt_prob_acc = []
        (h, c) = s
        if isinstance(x.data[0], int):
            x0 = torch.cat((Variable(torch.Tensor([0])), x))
            x1 = torch.cat((Variable(torch.Tensor([1])), x))
        if isinstance(x.data[0], float):
            x0 = torch.cat((Variable(torch.Tensor([0])).float(), x))
            x1 = torch.cat((Variable(torch.Tensor([1])).float(), x))

        # First iteration
        n = 0
        (h0, c0) = self.lstm(x1.view(1, 1, -1), (h.view(1, 1, -1), c.view(1, 1, -1)))[1]
        states.append(h0.view(-1))
        cells.append(c0.view(-1))
        outputs.append(self.fc_output(states[0]))
        halt_prob_acc.append(F.sigmoid(self.fc_halt(states[0])))
        halt_sum = halt_prob_acc[0].data[0]

        while halt_sum < 1-self.eps:
            n += 1
            (h0, c0) = self.lstm(x0.view(1, 1, -1), (states[n - 1].view(1, 1, -1), cells[n - 1].view(1, 1, -1)))[1]
            states.append(h0.view(-1))
            cells.append(c0.view(-1))
            outputs.append(self.fc_output(states[n]))
            halt_prob_acc.append(halt_prob_acc[n-1].clone() + F.sigmoid(self.fc_halt(states[n])))
            halt_sum = halt_prob_acc[n].data[0]

        outputs_tensor = torch.stack(outputs, dim=1)
        states_tensor = torch.stack(states, dim=1)
        cells_tensor = torch.stack(cells, dim=1)
        halt_prob_acc_tensor = torch.cat(halt_prob_acc)
        binarized_halt_tensor = self.f_bin(halt_prob_acc_tensor)

        output = torch.mv(outputs_tensor, binarized_halt_tensor)
        h = torch.mv(states_tensor, binarized_halt_tensor).view(-1)
        c = torch.mv(cells_tensor, binarized_halt_tensor).view(-1)
        ponder = torch.sum(1-binarized_halt_tensor)
        return output, (h, c), ponder


# ================================================ SkipACT =============================================================

class SkipARNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mu=2, num_layers=1, eps=0.01):
        super(SkipARNN, self).__init__()
        self.mu = mu
        self.eps = eps
        self.rnn = nn.RNN(1+input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)  # Add one to the input size for the binary flag
        self.fc_halt = nn.Linear(hidden_size, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([1.]))  # Set initial bias to avoid problems at the beginning
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.f_bin = Binarize(eps=eps)

    def forward(self, x, s0, y0, h0):
        outputs = []
        states = []
        halt_prob_acc = []
        if isinstance(x.data[0], int):
            x0 = torch.cat((Variable(torch.Tensor([0])), x))
            x1 = torch.cat((Variable(torch.Tensor([1])), x))
        if isinstance(x.data[0], float):
            x0 = torch.cat((Variable(torch.Tensor([0])).float(), x))
            x1 = torch.cat((Variable(torch.Tensor([1])).float(), x))

        # First iteration
        n = 0
        states.append(s0)
        outputs.append(y0)
        halt_prob_acc.append(h0)
        halt_sum = halt_prob_acc[0].data[0]

        if halt_sum < 1-self.eps:
            n += 1
            states.append(self.rnn(x1.view(1, 1, -1), states[n - 1].view(1, 1, -1))[0].view(-1))
            outputs.append(self.fc_output(states[n]))
            halt_prob_acc.append(halt_prob_acc[n - 1].clone() + self.mu*F.sigmoid(self.fc_halt(states[n])))
            halt_sum = halt_prob_acc[n].data[0]

        while halt_sum < 1-self.eps:
            n += 1
            states.append(self.rnn(x0.view(1, 1, -1), states[n-1].view(1, 1, -1))[0].view(-1))
            outputs.append(self.fc_output(states[n]))
            halt_prob_acc.append(halt_prob_acc[n-1].clone() + self.mu*F.sigmoid(self.fc_halt(states[n])))
            halt_sum = halt_prob_acc[n].data[0]

        h0_bin = self.f_bin(h0)
        if len(outputs) > 1:
            assert h0_bin.data[0] == 0
            outputs_tensor = torch.stack(outputs[1:], dim=1)
            states_tensor = torch.stack(states[1:], dim=1)
            halt_prob_acc_tensor = torch.cat(halt_prob_acc[1:])
            binarized_halt_tensor = self.f_bin(halt_prob_acc_tensor)

            output = h0_bin*y0 + (1-h0_bin)*torch.mv(outputs_tensor, binarized_halt_tensor)
            s = h0_bin*s0 + (1-h0_bin)*torch.mv(states_tensor, binarized_halt_tensor).view(-1)
            ponder = torch.sum(1-binarized_halt_tensor)
        else:
            assert h0_bin.data[0] == 1
            output = h0_bin*y0
            s = h0_bin*s0
            ponder = 0

        return output, s, ponder, halt_prob_acc[-1]-1
