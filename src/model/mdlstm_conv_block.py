import math
import torch
import torch.nn as nn

from model.mdlstm import MDLSTM
from utils.tensor_helper import to_best_device

from utils.tensor_helper import get_dim_out


class MDLSTMConvBlock(nn.Module):
    """
    Implementation of the architecture described in Handwriting Recognition with Large Multidimensional Long Short-Term Memory Recurrent Neural Networks
    by Paul V. et al.
    """
    def __init__(self, height: int, width: int, in_channels: int, out_lstm: int, out_conv: int, kernel: (int, int),
                 max_pool_kernel=(2, 2), dropout: float = 0.25):
        super(MDLSTMConvBlock, self).__init__()
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_conv, kernel_size=kernel)
        #self.norm = nn.BatchNorm2d(out_conv)
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel)
        h, w = get_dim_out(height, width)
        self.mdlstm = MDLSTM(height=h, width=w, in_channels=out_conv, out_channels=out_lstm)

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        self.mdlstm.initialize_weights()
        #self.norm.weight.data.fill_(1)

    def forward(self, x):
        #if self.dropout:
        #    x = self.dropout(x)
        x = self.conv(x)
        #x = self.norm(x)
        x = self.max_pool(x)
        x = torch.tanh(x)
        x = self.mdlstm(x)
        x = (x[:, 0, :, :, :] + x[:, 1, :, :, :] + x[:, 2, :, :, :] + x[:, 3, :, :, :]) / 4
        return x
        #return x[:, 0, :, :, :]
