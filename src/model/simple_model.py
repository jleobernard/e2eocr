from torch import nn

from model.my_lstm import CustomLSTM


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.lstm = CustomLSTM(input_sz=1, hidden_sz=11)
        self.lstm2 = CustomLSTM(input_sz=11, hidden_sz=11)

    def forward(self, x):
        """

        :param x: shape is (batch, channel, height, width)
        :return: shape is (batch, 10, width)
        """
        x = x.sum(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        return x

    def initialize_weights(self):
        self.lstm.init_weights()
        self.lstm2.init_weights()