import torch
import torch.nn as nn

from model.mdlstm import MDLSTM


class MDLSTMConvBlock(nn.Module):
    def __init__(self, height: int, width: int, in_channels: int, out_lstm: int, out_channels: int, kernel: (int, int), stride=2):
        super(MDLSTMConvBlock, self).__init__()
        self.mdlstm = MDLSTM(height=height, width=width, in_channels=in_channels, out_channels=out_lstm)
        self.conv_0 = nn.Conv2d(in_channels=out_lstm, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.conv_1 = nn.Conv2d(in_channels=out_lstm, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=out_lstm, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.conv_3 = nn.Conv2d(in_channels=out_lstm, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.mdlstm(x)
        x0 = self.conv_0(x[:, 0, :, :, :])
        x1 = self.conv_0(x[:, 1, :, :, :])
        x2 = self.conv_0(x[:, 2, :, :, :])
        x3 = self.conv_0(x[:, 3, :, :, :])
        return self.norm(torch.tanh(x0 + x1 + x2 + x3))
        #return torch.tanh(x0 + x1 + x2 + x3)
