import torch
import torch.nn as nn
from utils.utils_3d import psi


class InvertibleBottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0., affineBN=True, mult=2):
        """
        in_ch = 4
        out_ch = [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1] -> 12       out_ch, stride는 왼쪽에 있는 순서대로 가변적임
        stride = [2, 2, 2, 2, 8, 8, 8, 8, 8, 32, 32, 32] -> 12

        buid invertible bottleneck block
        """
        super(InvertibleBottleneckBlock, self).__init__()
        # False
        self.first = first
        # 1
        self.stride = stride
        #
        self.psi = psi(stride)
        '''
        bottlenect_block = layers = [
                                BatchNorm3d
                                ReLU
                                Conv3d
                                BatchNorm3d
                                ReLU
                                conv3d()
                                dropout
                                batchnorm3d
                                ReLU
                                conv3d
                            ]
        '''
        layers = []
        if not first:
            layers.append(nn.BatchNorm3d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        # ch는 무조건 1
        if int(out_ch//mult)==0:
            ch = 1
        else:
            ch =int(out_ch//mult)
        if self.stride ==2:
            layers.append(nn.Conv3d(in_ch // 2, ch, kernel_size=3,
                                    stride=(1,2,2), padding=1, bias=False))
        else:
            layers.append(nn.Conv3d(in_ch // 2, ch, kernel_size=3,
                                    stride=self.stride, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(ch, ch,
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm3d(ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(ch, out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        x = (x1, x2)
        return x