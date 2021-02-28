import torch.nn as nn
from model.mdlstm_conv_block import MDLSTMConvBlock
from utils.characters import characters

from utils.tensor_helper import get_dim_out

MAX_HIDDEN_UNITES = 120

class ParagraphReader(nn.Module):

    def __init__(self, height: int, width: int, feature_maps_multiplicity: int = 15, nb_layers: int = 3):
        super(ParagraphReader, self).__init__()
        mdlstm_conv_blocks = []
        h, w = height, width
        in_channels = 1
        dropout = 0.
        kernel = (3, 3)
        max_pool_kernel = (2, 2)
        for i in range(nb_layers):
            conv_maps, mdlstm_maps = self.get_feature_maps(i, feature_maps_multiplicity)
            if i == nb_layers - 1:
                mdlstm_maps = len(characters)
            mdlstm_conv_blocks.append(MDLSTMConvBlock(height=h, width=w, in_channels=in_channels,
                                                      out_lstm=mdlstm_maps, out_conv=conv_maps,
                                                      kernel=kernel, max_pool_kernel=max_pool_kernel, dropout=dropout))
            h, w = get_dim_out(h, w, kernel=kernel, max_pool_kernel=max_pool_kernel)
            in_channels = mdlstm_maps
            dropout = 0.25
        self.blocks = nn.ModuleList(mdlstm_conv_blocks)
        #self.dense = nn.Linear(in_features=mdlstm_maps, out_features=len(characters))

    def initialize_weights(self):
        for block in self.blocks:
            block.initialize_weights()

    def get_feature_maps(self, iteration: int, feature_maps_multiplicity: int):
        conv_maps = 15 if iteration == 0 else feature_maps_multiplicity * (2 * iteration + 1)
        return min(MAX_HIDDEN_UNITES, conv_maps), min(MAX_HIDDEN_UNITES, conv_maps + feature_maps_multiplicity)

    def forward(self, x):
        """

        :param x: Tensor of shape (batch, channel, height, width)
        :return: Tensor of shape (batch, sequence, len(characters))
        """
        batch_size, _, _, _ = x.shape
        for block in self.blocks:
            x = block(x)  # batch_size, in_channels, height, width = x.shape
        # Compress vertically
        x = x.sum(2)  # batch_size, in_channels, width = x.shape
        x = x.permute(0, 2, 1)  # batch_size, width, in_channels = x.shape
        #x = self.dense(x)  # batch_size, width, nb_chars = x.shape
        return x
