from torch import nn


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=11, num_layers=5, bidirectional=True, batch_first=True)

    def forward(self, x):
        """

        :param x: shape is (batch, channel, height, width)
        :return: shape is (batch, 10, width)
        """
        x = x.sum(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x

    def initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)