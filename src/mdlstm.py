import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from image_helper import get_dataset
import time


"""
https://link.springer.com/chapter/10.1007%2F978-3-540-74690-4_56
"""
class MDLSTMCellWeights(nn.Module):
    def __init__(self, entry_shape, out_nb_channels):
        super(MDLSTMCellWeights, self).__init__()
        in_nb_channels, height, weight = entry_shape
        x_parameters_shape = (out_nb_channels, in_nb_channels)
        h_parameters_shape = (out_nb_channels, out_nb_channels)
        bias_shape = (out_nb_channels, 1)
        self.w_ii = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_hi = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_i = nn.parameter.Parameter(torch.empty(bias_shape))

        self.w_if = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_hf = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_f = nn.parameter.Parameter(torch.empty(bias_shape))

        self.w_ig = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_hg = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_g = nn.parameter.Parameter(torch.empty(bias_shape))

        self.w_io = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_ho = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_o = nn.parameter.Parameter(torch.empty(bias_shape))

        # Weights of the weighted sum of the cs calculated for each direction
        self.weighted_sum = nn.parameter.Parameter(torch.rand(2))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.w_ii)
        torch.nn.init.xavier_uniform_(self.w_hi)
        torch.nn.init.xavier_uniform_(self.b_i)

        torch.nn.init.xavier_uniform_(self.w_if)
        torch.nn.init.xavier_uniform_(self.w_hf)
        torch.nn.init.xavier_uniform_(self.b_f)

        torch.nn.init.xavier_uniform_(self.w_ig)
        torch.nn.init.xavier_uniform_(self.w_hg)
        torch.nn.init.xavier_uniform_(self.b_g)

        torch.nn.init.xavier_uniform_(self.w_io)
        torch.nn.init.xavier_uniform_(self.w_ho)
        torch.nn.init.xavier_uniform_(self.b_o)


    def compute(self, x, c_prev_dim0, h_prev_dim0, c_prev_dim1, h_prev_dim1):
        """
        Computes the current activation of this cell as described in
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        and
        https://link.springer.com/chapter/10.1007%2F978-3-540-74690-4_56
        :param x: Current input
        :param c_prev_dim0: Previous state of the cell on the first dimension
        :param c_prev_dim1: Previous state of the cell on the second dimension
        :param h_prev_dim0: Hidden state of the previous cell on the first dimension
        :param h_prev_dim1: Hidden state of the previous cell on the second dimension
        :return:
        """
        xi = self.w_ii.matmul(x)
        xf = self.w_if.matmul(x)
        xg = self.w_ig.matmul(x)
        xo = self.w_io.matmul(x)
        print(f"Size of whi is {self.w_hi.shape} and that of h_prev_dim_0 is {h_prev_dim0.shape}")
        it_0 = torch.sigmoid(xi.add(self.w_hi.matmul(h_prev_dim0)).add(self.b_i))
        ft_0 = torch.sigmoid(xf.add(self.w_hf.matmul(h_prev_dim0)).add(self.b_f))
        gt_0 = torch.tanh(xg.add(self.w_hg.matmul(h_prev_dim0)).add(self.b_g))
        ot_0 = torch.sigmoid(xo.add(self.w_ho.matmul(h_prev_dim0)).add(self.b_o))
        ct0 = ft_0.mul(c_prev_dim0).add(it_0.mul(gt_0))
        ht0 = ot_0.mul(torch.tanh(ct0))

        it_1 = torch.sigmoid(xi.add(self.w_hi.matmul(h_prev_dim1)).add(self.b_i))
        ft_1 = torch.sigmoid(xf.add(self.w_hf.matmul(h_prev_dim1)).add(self.b_f))
        gt_1 = torch.tanh(xg.add(self.w_hg.matmul(h_prev_dim1)).add(self.b_g))
        ot_1 = torch.sigmoid(xo.add(self.w_ho.matmul(h_prev_dim1)).add(self.b_o))
        ct1 = ft_1.mul(c_prev_dim1).add(it_1.mul(gt_1))
        ht1 = ot_1.mul(torch.tanh(ct1))

        ct = torch.mul(ct0, self.weighted_sum[0]).add(torch.mul(ct1, self.weighted_sum[1]))
        ht = torch.mul(ht0, self.weighted_sum[0]).add(torch.mul(ht1, self.weighted_sum[1]))

        return ct, ht


class MDLSTMCell(nn.Module):
    def __init__(self, in_nb_channels, out_nb_channels):
        """
        :param out_nb_channels: Size of the embedding
        :param entry_shape: Entry shape
        """
        super(MDLSTMCell, self).__init__()
        x_parameters_shape = (in_nb_channels, out_nb_channels)
        h_parameters_shape = (out_nb_channels, out_nb_channels)
        bias_shape = (out_nb_channels)
        self.w_ii = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_hi = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_i = nn.parameter.Parameter(torch.empty(bias_shape))

        self.w_if = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_hf = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_f = nn.parameter.Parameter(torch.empty(bias_shape))

        self.w_ig = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_hg = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_g = nn.parameter.Parameter(torch.empty(bias_shape))

        self.w_io = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.w_ho = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b_o = nn.parameter.Parameter(torch.empty(bias_shape))

        # Weights of the weighted sum of the cs calculated for each direction
        self.weight_sum_1 = nn.parameter.Parameter(torch.rand(1))
        self.weight_sum_2 = nn.parameter.Parameter(torch.rand(1))

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.w_ii)
        torch.nn.init.xavier_uniform_(self.w_hi)
        torch.nn.init.xavier_uniform_(self.b_i)

        torch.nn.init.xavier_uniform_(self.w_if)
        torch.nn.init.xavier_uniform_(self.w_hf)
        torch.nn.init.xavier_uniform_(self.b_f)

        torch.nn.init.xavier_uniform_(self.w_ig)
        torch.nn.init.xavier_uniform_(self.w_hg)
        torch.nn.init.xavier_uniform_(self.b_g)

        torch.nn.init.xavier_uniform_(self.w_io)
        torch.nn.init.xavier_uniform_(self.w_ho)
        torch.nn.init.xavier_uniform_(self.b_o)

    def compute(self, x, c_prev_dim0, h_prev_dim0, c_prev_dim1, h_prev_dim1):
        """
        - For each output channel apply the same weights to each input channel
        - Sum the results
        :param x: Entry of shape (batch_size, in_nb_channels)
        :param c_prev_dim0:  previous state cell (batch_size, out_nb_channels) along the 1st dimension
        :param h_prev_dim0:  previous hidden state (batch_size, out_nb_channels) along the 1st dimension
        :param c_prev_dim1:  previous state cell (batch_size, out_nb_channels) along the 2nd dimension
        :param h_prev_dim1:  previous hidden state (batch_size, out_nb_channels) along the 2nd dimension
        :return: Tuple[c, h] each being of dimension (batch_size, out_nb_channels) which are the current state and hidden
        state for the current inputs
        """
        """
        print(f"x's shape is {x.shape}")
        print(f"h_prev_dim0's shape is {h_prev_dim0.shape}")
        print(f"h_prev_dim0[i, :]'s shape is {h_prev_dim0[0,:].shape}")
        """
        """
        Computes the current activation of this cell as described in
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        and
        https://link.springer.com/chapter/10.1007%2F978-3-540-74690-4_56
        """
        xi = x @ self.w_ii
        xf = x @ self.w_if
        xg = x @ self.w_ig
        xo = x @ self.w_io
        it_0 = torch.sigmoid(xi.add(h_prev_dim0 @ self.w_hi).add(self.b_i))
        ft_0 = torch.sigmoid(xf.add(h_prev_dim0 @ self.w_hf).add(self.b_f))
        gt_0 = torch.tanh(xg.add(h_prev_dim0 @ self.w_hg).add(self.b_g))
        ot_0 = torch.sigmoid(xo.add(h_prev_dim0 @ self.w_ho).add(self.b_o))
        ct0 = ft_0.mul(c_prev_dim0).add(it_0.mul(gt_0))
        ht0 = ot_0.mul(torch.tanh(ct0))

        it_1 = torch.sigmoid(xi.add(h_prev_dim1 @ self.w_hi).add(self.b_i))
        ft_1 = torch.sigmoid(xf.add(h_prev_dim1 @ self.w_hf).add(self.b_f))
        gt_1 = torch.tanh(xg.add(h_prev_dim1 @ self.w_hg).add(self.b_g))
        ot_1 = torch.sigmoid(xo.add(h_prev_dim1 @ self.w_ho).add(self.b_o))
        ct1 = ft_1.mul(c_prev_dim1).add(it_1.mul(gt_1))
        ht1 = ot_1.mul(torch.tanh(ct1))

        # TODO fix this
        ct = ct0 * self.weight_sum_1 + ct1 * self.weight_sum_2
        ht = ht0 * self.weight_sum_1 + ht1 * self.weight_sum_2

        return ct, ht


class MDLSTM(nn.Module):
    def __init__(self, height: int, width: int, in_nb_channels: int, out_nb_channels: int):
        super(MDLSTM, self).__init__()
        self.out_nb_channels = out_nb_channels
        self.width = width
        self.height = height
        area = width * height
        # One LSTM per direction
        entry_shape = (in_nb_channels, self.height, self.width)
        self.lstm_lr_tb = MDLSTMCell(in_nb_channels=in_nb_channels, out_nb_channels=out_nb_channels)
        self.lstm_rl_tb = MDLSTMCell(in_nb_channels=in_nb_channels, out_nb_channels=out_nb_channels)
        self.lstm_lr_bt = MDLSTMCell(in_nb_channels=in_nb_channels, out_nb_channels=out_nb_channels)
        self.lstm_rl_bt = MDLSTMCell(in_nb_channels=in_nb_channels, out_nb_channels=out_nb_channels)
        # Indices in the order the lstms need to scan entries
        self.indices_lr_tb = self.to_coordinates(np.arange(start=0, stop=area, step=1))
        self.indices_rl_tb = self.to_coordinates(np.concatenate([np.arange((x + 1) * width - 1, x * width - 1, step = -1) for x in range(height)]))
        self.indices_lr_bt = self.to_coordinates(np.concatenate([np.arange(x * width, (x + 1) * width, step=1) for x in range(height - 1, -1,  -1)]))
        self.indices_rl_bt = self.to_coordinates(np.arange(start=area - 1, stop=-1, step=-1))

    def to_coordinates(self, indices: list):
        return [(self.to_y(idx), self.to_x(idx)) for idx in indices]

    def to_y(self, idx):
        return idx // self.width

    def to_x(self, idx):
        return idx % self.width

    def is_dim_0_out(self, y: int, delta: int):
        return y <= 0 if delta < 0 else y >= self.height - 1

    def is_dim_1_out(self, x: int, delta: int):
        return x <= 0 if delta < 0 else x >= self.width - 1

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of size (batch_size, in_nb_channels, height, width)
        :return: Tensor of size (batch_size, 4, out_channels, height, width)
        """
        batch_size, in_nb_channels, height, width = x.shape
        # Hidden states will be the output of the function
        hidden_states = torch.empty((batch_size, 4, self.out_nb_channels, height, width), requires_grad=False)
        cell_states = torch.empty((batch_size, 4, self.out_nb_channels, height, width), requires_grad=False)
        # For each direction we're going to compute hidden_states and their activations
        # Side note : This could be processed in parallel
        params = [{"indices": self.indices_lr_tb, "prev": (-1, -1), "lstm": self.lstm_lr_tb},
                  {"indices": self.indices_rl_tb, "prev": (-1, 1), "lstm": self.lstm_rl_tb},
                  {"indices": self.indices_lr_bt, "prev": (1, -1), "lstm": self.lstm_lr_bt},
                  {"indices": self.indices_rl_bt, "prev": (1, 1), "lstm": self.lstm_rl_bt}]
        direction = 0
        for param in params:
            prev = param["prev"]
            lstm = param["lstm"]
            for idx in param["indices"]:
                y_height, x_width = idx
                delta_0, delta_1 = prev
                # If we're on the first row the previous element is the vector of the good shape with 0s
                if self.is_dim_0_out(y_height, delta_0):
                    prev_0_c = torch.zeros((batch_size, self.out_nb_channels))
                    prev_0_h = torch.zeros((batch_size, self.out_nb_channels))
                else:
                    # Otherwise we get back the previously computed c and h for this direction and coordinates
                    # So the tensors are of the shape (batch_size, out_nb_channels)
                    prev_0_c = cell_states[:, direction, :, y_height + delta_0, x_width]
                    prev_0_h = hidden_states[:, direction, :, y_height + delta_0, x_width]
                # If we're on the first column the previous element is the vector of the good shape with 0s
                if self.is_dim_1_out(x_width, delta_1):
                    prev_1_c = torch.zeros((batch_size, self.out_nb_channels))
                    prev_1_h = torch.zeros((batch_size, self.out_nb_channels))
                else:
                    # Otherwise we get back the previously computed c and h for this direction and coordinates
                    # So the tensors are of the shape (batch_size, out_nb_channels)
                    prev_1_c = cell_states[:, direction, :, y_height, x_width + delta_1]
                    prev_1_h = hidden_states[:, direction, :, y_height, x_width + delta_1]
                # The current input is a tensor of shape (batch_size, input_channels) at coordinates (x,y)
                current_input = x[:, :, y_height, x_width]
                cs, hs = lstm.compute(current_input, prev_0_c, prev_0_h, prev_1_c, prev_1_h)
                hidden_states[:, direction, :, y_height, x_width] = hs
                cell_states[:, direction, :, y_height, x_width] = cs
            direction += 1
        return hidden_states
"""
ds = get_dataset('data/00-alphabet', width=50, height=50)
dataloader = DataLoader(ds, batch_size=10, shuffle=False)
model = MDLSTM(height=50, width=50, in_nb_channels=1, out_nb_channels=3)
start = time.time()
for i, batch_data in enumerate(dataloader):
    data, labels = batch_data
    model.forward(data)
    print(f"Batch {i} done")
end = time.time()
print(f"It took {end - start}")
"""