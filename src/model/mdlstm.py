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
        self.b = nn.parameter.Parameter(torch.empty(bias_shape))

        # Weights of the weighted sum of the cs calculated for each direction
        #self.weight_sum_1 = nn.parameter.Parameter(to_best_device(torch.rand(1)))
        #self.weight_sum_2 = nn.parameter.Parameter(to_best_device(torch.rand(1)))
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.w0)
        torch.nn.init.xavier_uniform_(self.u0)
        torch.nn.init.xavier_uniform_(self.w1)
        torch.nn.init.xavier_uniform_(self.u1)
        torch.nn.init.uniform_(self.b)

    def compute(self, x, c_prev_dim0, h_prev_dim0, c_prev_dim1, h_prev_dim1):
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

        #ct = ct0 * self.weight_sum_1 + ct1 * self.weight_sum_2
        #ht = ht0 * self.weight_sum_1 + ht1 * self.weight_sum_2

        #return ct, ht


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
        # Indices in the order the lstms need to scan entries
        self.indices_lr_tb = np.arange(start=0, stop=area, step=1)
        self.indices_rl_tb = np.concatenate([np.arange((x + 1) * width - 1, x * width - 1, step = -1) for x in range(height)])
        self.indices_lr_bt = np.concatenate([np.arange(x * width, (x + 1) * width, step=1) for x in range(height - 1, -1,  -1)])
        self.indices_rl_bt = np.arange(start=area - 1, stop=-1, step=-1)

        self.params = [{"indices": self.indices_lr_tb, "prev": (-1, -1), "lstm": self.lstm_lr_tb},
                  {"indices": self.indices_rl_tb, "prev": (-1, 1), "lstm": self.lstm_rl_tb},
                  {"indices": self.indices_lr_bt, "prev": (1, -1), "lstm": self.lstm_lr_bt},
                  {"indices": self.indices_rl_bt, "prev": (1, 1), "lstm": self.lstm_rl_bt}]
        self.fold = torch.nn.Fold(output_size=(self.height, self.width), kernel_size=(1, 1))

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

    def reorder_for_direction(self, tensor_list: List[torch.Tensor], direction: List[int], desired_shape: tuple) -> torch.Tensor:
        """
        Reorder computed hidden states of the tensor list so it matches the left->right/top->bottom order

        :param tensor_list: List of tensors, each containing the hidden state/cell state of a pixel
        :param direction: list of index swaps
        :param desired_shape: The shape of the output
        :return: A tensor with the desired shape
        """
        tensor = to_best_device(torch.stack(tensor_list))
        tensor = torch.index_select(tensor, 0, to_best_device(torch.tensor(direction)))
        tensor = tensor.permute(1, 2, 0)
        result = tensor.reshape(desired_shape)
        return result

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
        # Side note : This could be processed in parallel
        global_hidden_states = len(self.params) * [None]
        streams = [torch.cuda.Stream() for _ in self.params] if cuda_available else []
        if cuda_available:
            torch.cuda.synchronize()
        for i, param in enumerate(self.params):
            x_ordered = self.flipped_image(x, direction=i)
            if cuda_available:
                stream = streams[i]
                with torch.cuda.stream(stream):
                    hidden_states_direction = self.do_forward(x_ordered, param)
            else:
                hidden_states_direction = self.do_forward(x_ordered, param)
            global_hidden_states[i] = hidden_states_direction
        if cuda_available:
            torch.cuda.synchronize()
        # Needs to be transposed because we stacked by direction while we expect the first dimension to be batch
        stacked = torch.stack(global_hidden_states)
        return to_best_device(stacked.transpose(0, 1))

    def do_forward(self, x, param):
        batch_size, in_channels, height, width = x.shape
        lstm = param["lstm"]
        hidden_states_direction = []
        cell_states_direction = []
        i = 0
        coordinates = self.to_coordinates(self.params[0]["indices"])
        for idx in coordinates:
            y_height, x_width = idx
            # If we're on the first row the previous element is the vector of the good shape with 0s
            if i - width < 0:
                prev_0_c = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                prev_0_h = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
            else:
                # Otherwise we get back the previously computed c and h for this direction and coordinates
                # So the tensors are of the shape (batch_size, out_channels)
                idx_to_prev = i - width
                prev_0_c = cell_states_direction[idx_to_prev]
                prev_0_h = hidden_states_direction[idx_to_prev]
            # If we're on the first column the previous element is the vector of the good shape with 0s
            if i % width == 0:
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
            cs, hs = lstm.compute(current_input, prev_0_c, prev_0_h, prev_1_c, prev_1_h)
            cell_states_direction.append(cs)
            hidden_states_direction.append(hs)
            i += 1
        return self.fold(torch.stack(hidden_states_direction, dim=2))
