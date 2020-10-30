import torch
import torch.nn as nn
from torch.autograd import Variable
from RPM.STConvLSTM import STConvLSTMCell


class ReversiblePredictor(nn.Module):
    def __init__(self, input_size, hidden_size,output_size,n_layers,batch_size,temp =3, w =8,h = 8):
        super(ReversiblePredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.temp = temp
        self.w = w
        self.h = h

        self.convlstm = nn.ModuleList(
                [STConvLSTMCell(input_size, hidden_size,hidden_size) if i == 0 else STConvLSTMCell(hidden_size,hidden_size, hidden_size) for i in
                 range(self.n_layers)])

        self.att = nn.ModuleList([nn.Sequential(nn.Conv3d(self.hidden_size, self.hidden_size, 1, 1, 0),
                                                # nn.ReLU(),
                                                # nn.Conv3d(self.hidden_size, self.hidden_size, 3, 1, 1),
                                                nn.Sigmoid()
                                                ) for i in range(self.n_layers)])

        self.hidden = self.init_hidden()
        self.prev_hidden = self.hidden

    def init_hidden(self):
        hidden = []

        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size,self.temp,self.w,self.h).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size,self.temp,self.w,self.h).cuda())))
        return hidden

    def forward(self, input):
        input_, memo = input
        x1, x2 = input_
        for i in range(self.n_layers):
            out = self.convlstm[i]((x1,memo), self.hidden[i])
            self.hidden[i] = out[0]
            memo = out[1]
            g = self.att[i](self.hidden[i][0])
            x2 = (1 - g) * x2 + g * self.hidden[i][0]
            x1, x2 = x2, x1

        return (x1, x2), memo
