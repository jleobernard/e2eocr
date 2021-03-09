import torch.nn as nn
import torch.nn.init as init

from utils.characters import characters, nb_characters


class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3))
        self.cnn1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.cnn5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.cnn6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2))
        self.norm512 = nn.BatchNorm2d(512, affine=False)
        self.lstm = nn.LSTM(batch_first=True, bidirectional=True, num_layers=2, input_size=512, hidden_size=256)
        self.lstm_out = nn.LSTM(batch_first=True, bidirectional=True, num_layers=1, input_size=256, hidden_size=len(characters))
        self.max_pool22 = nn.MaxPool2d(kernel_size=2)
        self.max_pool12 = nn.MaxPool2d(kernel_size=(1, 2))


    def initialize_weights(self):
        nn.init.xavier_uniform_(self.cnn0.weight)
        nn.init.xavier_uniform_(self.cnn1.weight)
        nn.init.xavier_uniform_(self.cnn2.weight)
        nn.init.xavier_uniform_(self.cnn3.weight)
        nn.init.xavier_uniform_(self.cnn4.weight)
        nn.init.xavier_uniform_(self.cnn5.weight)
        nn.init.xavier_uniform_(self.cnn6.weight)
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for param in self.lstm_out.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

    def forward(self, x):
        """

        :param x: Tensor of shape (batch, channel, height, width)
        :return: Tensor of shape (batch, sequence, len(characters))
        """
        batch_size, _, _, _ = x.shape
        x = self.max_pool22(nn.functional.relu(self.cnn0(x)))
        x = self.max_pool22(nn.functional.relu(self.cnn1(x)))
        x = nn.functional.relu(self.cnn2(x))
        x = nn.functional.relu(self.cnn3(x))
        x = self.max_pool12(x)
        x = nn.functional.relu(self.cnn4(x))
        x = self.norm512(x)
        x = nn.functional.relu(self.cnn5(x))
        x = self.norm512(x)
        x = self.max_pool12(x)
        x = nn.functional.relu(self.cnn6(x))
        # Compress vertically
        x = x.sum(2)  # batch_size, in_channels, width = x.shape
        x = x.permute(0, 2, 1)  # batch_size, width, in_channels = x.shape
        x, _ = self.lstm(x)
        x = (x[:, :, :256] + x[:, :, 256:]) / 2
        x, _ = self.lstm_out(x)
        x = (x[:, :, :nb_characters] + x[:, :, nb_characters:]) / 2
        return x
