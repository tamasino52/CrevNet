import torch
import torch.nn as nn
from utils.utils_3d import merge, psi
from AutoEncoder.InvertibleBottleneckBlock import InvertibleBottleneckBlock


class AutoEncoder(nn.Module):
    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2, dropout_rate=0., affineBN=True, in_shape=None, mult=2):
        """
        :param nBlocks: [4, 5, 3]
        :param nStrides: [1, 2, 2]
        :param nChannels: None
        :param init_ds: 2
        :param dropout_rate: 0
        :param affineBN: True
        :param in_shape: [1, 64, 64]
        :param mult: 2
        """

        super(AutoEncoder, self).__init__()
        # 64 // (2 ** (2 + 1)) = 64 // 8 = 8
        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + init_ds // 2)
        # 2
        self.init_ds = init_ds
        # 1 * 2 ** 2 = 4
        self.in_ch = in_shape[0] * 2 ** self.init_ds
        # [4, 5, 3]
        self.nBlocks = nBlocks
        self.first = True

        # print('')
        # print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3))
        if not nChannels:
            # [2, 8, 32, 128]
            nChannels = [self.in_ch // 2, self.in_ch // 2 * 4,
                         self.in_ch // 2 * 4 ** 2, self.in_ch // 2 * 4 ** 3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(InvertibleBottleneckBlock, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """
        :param _block: irevnet_blck - initialize lately
        :param nChannels: [2, 8, 32, 128]
        :param nBlocks: [4, 5, 3]
        :param nStrides: [1, 2, 2]
        :param dropout_rate: 0
        :param affineBN: True
        :param in_ch: 4
        :param mult: 2
        :return:
        """

        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            '''
            channel = 2, depth = 4, stride = 1
            strides = [1, 1, 1, 1]
            channels = [2, 2, 2, 2]

            channel = 8, depth = 5, stride = 2
            strides = [1, 1, 1, 1, 2, 1, 1, 1, 1]
            channels = [2, 2, 2, 2, 8, 8, 8, 8, 8]

            channel = 32, depth = 3, stride = 2
            strides = [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1] -> 12
            channels = [2, 2, 2, 2, 8, 8, 8, 8, 8, 32, 32, 32] -> 12
            '''
            strides = strides + ([stride] + [1] * (depth - 1))
            channels = channels + ([channel] * depth)

        # 총 12번 돌음
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, input, is_predict=True):

        if is_predict:
            n = self.in_ch // 2
            if self.init_ds != 0:
                x = self.init_psi.forward(input)
            out = (x[:, :n, :, :, :], x[:, n:, :, :, :])

            for block in self.stack:
                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = merge(out[0], out[1])
            x = self.init_psi.inverse(out)
        return x