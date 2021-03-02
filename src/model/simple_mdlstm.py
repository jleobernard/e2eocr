from torch import nn

from model.mdlstm import MDLSTM


class SimpleModelMDLSTM(nn.Module):

    def __init__(self, width=5):
        super(SimpleModelMDLSTM, self).__init__()
        self.norm_1 = nn.BatchNorm2d(1, affine=False)
        self.norm_11 = nn.BatchNorm2d(11, affine=False)
        self.mdlstm0 = MDLSTM(in_channels=1, out_channels=11, height=1, width=width)
        self.mdlstm1 = MDLSTM(in_channels=11, out_channels=11, height=1, width=width)

    def forward(self, x):
        """

        :param x: shape is (batch, channel, height, width)
        :return: shape is (batch, 10, width)
        """
        x = x.sum(2).unsqueeze(2)
        x = self.mdlstm0(x)
        x = x.sum(1)
        x = self.mdlstm1(x)
        x = x.sum(1)
        x = x.squeeze(dim=2)
        return x

    def initialize_weights(self):
        self.mdlstm0.initialize_weights()
        self.mdlstm1.initialize_weights()