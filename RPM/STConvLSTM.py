import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class STConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, memo_size):
        super(STConvLSTMCell, self).__init__()
        self.KERNEL_SIZE = 3
        self.PADDING = self.KERNEL_SIZE // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memo_size = memo_size
        self.in_gate = nn.Conv3d(input_size + hidden_size + hidden_size , hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate = nn.Conv3d(input_size + hidden_size + hidden_size , hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate = nn.Conv3d(input_size + hidden_size + hidden_size, hidden_size , self.KERNEL_SIZE, padding=self.PADDING)

        self.in_gate1 = nn.Conv3d(input_size + memo_size + hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate1 = nn.Conv3d(input_size + memo_size + hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate1 = nn.Conv3d(input_size + memo_size + hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.w1 = nn.Conv3d(hidden_size + memo_size, hidden_size, 1)
        self.out_gate = nn.Conv3d(input_size + hidden_size + hidden_size + memo_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)

    def forward(self, input, prev_state):
        input_, prev_memo = input
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden, prev_cell), 1)

        in_gate = F.sigmoid(self.in_gate(stacked_inputs))
        remember_gate = F.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = F.tanh(self.cell_gate(stacked_inputs))

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo, cell), 1)

        in_gate1 = F.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = F.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = F.tanh(self.cell_gate1(stacked_inputs1))

        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = F.sigmoid(self.out_gate(torch.cat((input_, prev_hidden, cell, memo), 1)))
        hidden = out_gate * F.tanh(self.w1(torch.cat((cell, memo), 1)))

        return (hidden, cell), memo
