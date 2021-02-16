import math
import torch
import torch.nn as nn

from mdlstm import MDLSTM
from mdlstm_conv_block import MDLSTMConvBlock


class ParagraphReader(nn.Module):
    def __init__(self, height: int, width: int):
        super(ParagraphReader, self).__init__()
        n_dense = 100
        self.block0 = MDLSTMConvBlock(height=height, width=width, in_channels=1, out_lstm=2, out_channels=6, kernel=(2, 4))
        h, w = self.get_dim_out(height, width)
        self.block1 = MDLSTMConvBlock(height=h, width=w, in_channels=6, out_lstm=10, out_channels=20, kernel=(2, 4))
        h, w = self.get_dim_out(h, w)
        self.block1 = MDLSTMConvBlock(height=h, width=w, in_channels=20, out_lstm=30, out_channels=50, kernel=(2, 4))
        h, w = self.get_dim_out(h, w)
        self.mdlstm = MDLSTM(height=h, width=w, in_channels=20, out_channels=50)
        self.dense0 = nn.Linear(in_features=50, out_features=n_dense)
        self.dense1 = nn.Linear(in_features=50, out_features=n_dense)
        self.dense2 = nn.Linear(in_features=50, out_features=n_dense)
        self.dense3 = nn.Linear(in_features=50, out_features=n_dense)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.mdlstm(x)
        # à partir d'ici ça bloque parce que les dimensions ne sont pas bonnes (batch, 50, 16, 14) VS (_, 50, 100)
        x0 = self.dense0(x[:, 0, :, :, :])
        x1 = self.dense1(x[:, 1, :, :, :])
        x2 = self.dense2(x[:, 2, :, :, :])
        x3 = self.dense3(x[:, 3, :, :, :])
        final_x = x0 + x1 + x2 + x3
        # Add CTC here
        return final_x

    def get_dim_out(self, height: int, width: int, kernel: (int, int) = (2, 4), stride: int = 2, padding=0, dilatation=1):
        """
        Applies the rules described at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=convolution#torch.nn.Conv2d
        :param height:
        :param width:
        :param stride:
        :param kernel:
        :param padding:
        :param dilatation:
        :return:
        """
        hout = math.floor((height + 2 * padding - dilatation * (kernel[0] - 1) - 1) / stride)
        wout = math.floor((width + 2 * padding - dilatation * (kernel[1] - 1) - 1) / stride)
        return hout, wout
