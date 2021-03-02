import torch.nn as nn

from model.mdlstm import MDLSTM
from model.mdlstm_conv_block import MDLSTMConvBlock
from utils.characters import characters

from utils.tensor_helper import get_dim_out

MAX_HIDDEN_UNITES = 120

class MultiMdlstm(nn.Module):

    def __init__(self, height: int, width: int, feature_maps_multiplicity: int = 15, nb_layers: int = 5, nb_classes: int = -1):
        super(MultiMdlstm, self).__init__()
        mdlstms = []
        in_channels = 1
        for i in range(nb_layers):
            hidden_size = (i + 1) * feature_maps_multiplicity
            if i == nb_layers - 1:
                hidden_size = nb_classes if nb_classes > 0 else len(characters)
            mdlstms.append(MDLSTM(height=height, width=width, in_channels=in_channels, out_channels=hidden_size))
            in_channels = hidden_size
        self.mdlstms = nn.ModuleList(mdlstms)

    def initialize_weights(self):
        for block in self.mdlstms:
            block.initialize_weights()

    def forward(self, x):
        """

        :param x: Tensor of shape (batch, channel, height, width)
        :return: Tensor of shape (batch, sequence, len(characters))
        """
        batch_size, _, _, _ = x.shape
        for block in self.mdlstms:
            x = block(x)  # batch_size, 4, in_channels, height, width = x.shape
            x = x.sum(1) / 4  # batch_size, in_channels, height, width = x.shape
        # Compress vertically
        x = x.sum(2)  # batch_size, in_channels, width = x.shape
        return x
