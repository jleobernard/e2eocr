import torch.nn as nn
from model.mdlstm_conv_block import MDLSTMConvBlock
from utils.characters import characters, nb_characters

from utils.tensor_helper import get_dim_out


class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
        cnns = []
        self.norms = []
        in_channels = 1
        for i in range(3):
            out_channels = (i + 1) * 32
            cnns.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2)))
            in_channels = out_channels
            self.norms.append(nn.BatchNorm2d(out_channels, affine=False))
        self.cnns = nn.ModuleList(cnns)
        self.lstm = nn.LSTM(batch_first=True, bidirectional=True, num_layers=5, input_size=out_channels, hidden_size=len(characters))
        self.max_pool = nn.MaxPool2d(kernel_size=2)


    def initialize_weights(self):
        for cnn in self.cnns:
            nn.init.xavier_uniform_(cnn.weight)

    def forward(self, x):
        """

        :param x: Tensor of shape (batch, channel, height, width)
        :return: Tensor of shape (batch, sequence, len(characters))
        """
        batch_size, _, _, _ = x.shape
        for i, cnn in enumerate(self.cnns):
            x = cnn(x)
            x = nn.functional.relu(x)
            x = self.max_pool(x)
            x = self.norms[i](x)
        # Compress vertically
        x = x.sum(2)  # batch_size, in_channels, width = x.shape
        x = x.permute(0, 2, 1)  # batch_size, width, in_channels = x.shape
        x, _ = self.lstm(x)
        x = (x[:, :, :nb_characters] + x[:, :, nb_characters:]) / 2
        return x
