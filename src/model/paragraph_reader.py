import math
import torch.nn as nn
from model.mdlstm_conv_block import MDLSTMConvBlock


# Size of vocabulary is
# - 1 blank
# - 26 unaccentuated letters
# - 10 numbers
SIZE_VOC = 1 + 26 + 10
OUT_CHANNELS_LAST_CNN = 50

class ParagraphReader(nn.Module):
    def __init__(self, height: int, width: int):
        super(ParagraphReader, self).__init__()
        self.block0 = MDLSTMConvBlock(height=height, width=width, in_channels=1, out_lstm=2, out_channels=6, kernel=(2, 4))
        h, w = self.get_dim_out(height, width)
        self.block1 = MDLSTMConvBlock(height=h, width=w, in_channels=6, out_lstm=10, out_channels=20, kernel=(2, 4))
        h, w = self.get_dim_out(h, w)
        self.block2 = MDLSTMConvBlock(height=h, width=w, in_channels=20, out_lstm=30, out_channels=OUT_CHANNELS_LAST_CNN, kernel=(2, 4))
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.lstm = nn.LSTM(input_size=OUT_CHANNELS_LAST_CNN, hidden_size=50, batch_first=True)
        self.dense = nn.Linear(in_features=50, out_features=SIZE_VOC)

    def forward(self, x):
        #print(f"Shape at the beginning is {x.shape}")
        batch_size, _, _, _ = x.shape
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        #print(f"Shape before is {x.shape}")
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size, height * width, OUT_CHANNELS_LAST_CNN)
        #print(f"Shape after is {x.shape}")
        x, _ = self.lstm(x)
        #print(f"Shape after LSTM is {x.shape}")
        x = self.dense(x)
        return x

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
