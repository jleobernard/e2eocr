import torch
import torch.nn as nn
import numpy as np
from typing import List

from utils.tensor_helper import to_best_device, cuda_available

"""
https://link.springer.com/chapter/10.1007%2F978-3-540-74690-4_56
"""
class MDLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param out_channels: Size of the embedding
        :param entry_shape: Entry shape
        """
        super(MDLSTMCell, self).__init__()
        self.out_channels = out_channels
        x_parameters_shape = (in_channels, out_channels * 5)
        h_parameters_shape = (out_channels, out_channels * 5)
        bias_shape = (out_channels * 5)
        self.w0 = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.u0 = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.w1 = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.u1 = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b = nn.parameter.Parameter(torch.zeros(bias_shape))

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.w0)
        torch.nn.init.xavier_uniform_(self.u0)
        torch.nn.init.xavier_uniform_(self.w1)
        torch.nn.init.xavier_uniform_(self.u1)
        #torch.nn.init.uniform_(self.b)

    def forward(self, x, c_prev_dim0, h_prev_dim0, c_prev_dim1, h_prev_dim1):
        """
        - For each output channel apply the same weights to each input channel
        - Sum the results
        :param x: Entry of shape (batch_size, in_channels)
        :param c_prev_dim0:  previous state cell (batch_size, out_channels) along the 1st dimension
        :param h_prev_dim0:  previous hidden state (batch_size, out_channels) along the 1st dimension
        :param c_prev_dim1:  previous state cell (batch_size, out_channels) along the 2nd dimension
        :param h_prev_dim1:  previous hidden state (batch_size, out_channels) along the 2nd dimension
        :return: Tuple[c, h] each being of dimension (batch_size, out_channels) which are the current state and hidden
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
        gates_0 = x @ self.w0 + h_prev_dim0 @ self.u0 + h_prev_dim1 @ self.u1 + self.b
        oc = self.out_channels
        it_0 = torch.sigmoid(gates_0[: , :oc])
        ft_0 = torch.sigmoid(gates_0[: , oc:oc*2])
        gt_0 = torch.tanh(gates_0[: , oc*2:oc*3])
        ot_0 = torch.sigmoid(gates_0[: , oc*3:oc*4])
        lt_0 = torch.sigmoid(gates_0[: , oc*4:]) # The lambda gte
        ct0 = ft_0 * ((lt_0 * c_prev_dim0) + ((1 - lt_0) * c_prev_dim1)) + it_0 * gt_0
        ht0 = ot_0 * torch.tanh(ct0)

        return ct0, ht0


class MDLSTM(nn.Module):
    def __init__(self, height: int, width: int, in_channels: int, out_channels: int):
        super(MDLSTM, self).__init__()
        self.out_channels = out_channels
        self.width = width
        self.height = height
        area = width * height
        # One LSTM per direction
        self.lstm_lr_tb = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.lstm_rl_tb = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.lstm_lr_bt = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.lstm_rl_bt = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.params = [self.lstm_lr_tb, self.lstm_rl_tb, self.lstm_lr_bt, self.lstm_rl_bt]
        self.fold = torch.nn.Fold(output_size=(self.height, self.width), kernel_size=(1, 1))

    def initialize_weights(self):
        self.lstm_lr_tb.initialize_weights()
        self.lstm_rl_tb.initialize_weights()
        self.lstm_lr_bt.initialize_weights()
        self.lstm_rl_bt.initialize_weights()

    def flipped_image(self, x: torch.Tensor, direction: int):
        if direction == 0: # LRTP
            return x
        elif direction == 1: # RLTB
            return torch.flip(x, (3,))
        elif direction == 2: # LRBT
            return torch.flip(x, (2,))
        elif direction == 3: # RLBT
            return torch.flip(x, (2, 3,))

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of size (batch_size, in_channels, height, width)
        :return: Tensor of size (batch_size, 4, out_channels, height, width)
        """
        # For each direction we're going to compute hidden_states and their activations
        global_hidden_states = len(self.params) * [None]
        streams = [torch.cuda.Stream() for _ in self.params] if cuda_available else []
        if cuda_available:
            torch.cuda.synchronize()
        for i, lstm in enumerate(self.params):
            x_ordered = self.flipped_image(x, direction=i)
            if cuda_available:
                stream = streams[i]
                with torch.cuda.stream(stream):
                    hidden_states_direction = self.do_forward(x_ordered, lstm)
            else:
                hidden_states_direction = self.do_forward(x_ordered, lstm)
            global_hidden_states[i] = self.flipped_image(hidden_states_direction, direction=i)
        if cuda_available:
            torch.cuda.synchronize()
        # Each element in global_hidden_states is of shape (batch, channel, height, width)
        # Needs to be transposed because we stacked by direction while we expect the first dimension to be batch
        return torch.stack(global_hidden_states, dim=1) # (batch, 4, channel, height, width) = stacked.shape

    def do_forward(self, x, lstm):
        batch_size, in_channels, height, width = x.shape
        hidden_states_direction = []
        cell_states_direction = []
        i = -1
        for y_height in range(height):
            for x_width in range(width):
                i += 1
                # If we're on the first row the previous element is the vector of the good shape with 0s
                if y_height == 0:
                    prev_0_c = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                    prev_0_h = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                else:
                    # Otherwise we get back the previously computed c and h for this direction and coordinates
                    # So the tensors are of the shape (batch_size, out_channels)
                    idx_to_prev = i - width
                    prev_0_c = cell_states_direction[idx_to_prev]
                    prev_0_h = hidden_states_direction[idx_to_prev]
                # If we're on the first column the previous element is the vector of the good shape with 0s
                if x_width == 0:
                    prev_1_c = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                    prev_1_h = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                else:
                    # Otherwise we get back the previously computed c and h for this direction and coordinates
                    # So the tensors are of the shape (batch_size, out_channels)
                    idx_to_prev = i - 1
                    prev_1_c = cell_states_direction[idx_to_prev]
                    prev_1_h = hidden_states_direction[idx_to_prev]
                # The current input is a tensor of shape (batch_size, input_channels) at coordinates (x,y)
                current_input = x[:, :, y_height, x_width]
                cs, hs = lstm(current_input, prev_0_c, prev_0_h, prev_1_c, prev_1_h)
                cell_states_direction.append(cs)
                hidden_states_direction.append(hs)
        # Check the next line
        return self.fold(torch.stack(hidden_states_direction, dim=2))
