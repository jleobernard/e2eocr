import math
import torch
import torch.nn as nn
import sys
import time
sys.path.insert(1, '/opt/projetcs/ich/e2eocr/src')
from model.mdlstm import MDLSTM, MDLSTMCell

# in_channels=2, out_channels=3
weights = [
    {
        "w_ii": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.]])),
        "w_hi": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])),
        "b_i": nn.parameter.Parameter(torch.tensor([9., 10., 11.])),

        "w_if": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.]])),
        "w_hf": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])),
        "b_f": nn.parameter.Parameter(torch.tensor([9., 10., 11.])),

        "w_ig": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.]])),
        "w_hg": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])),
        "b_g": nn.parameter.Parameter(torch.tensor([9., 10., 11.])),

        "w_io": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.]])),
        "w_ho": nn.parameter.Parameter(torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])),
        "b_o": nn.parameter.Parameter(torch.tensor([9., 10., 11.])),

        "weight_sum_1": nn.parameter.Parameter(torch.tensor([1.])),
        "weight_sum_2": nn.parameter.Parameter(torch.tensor([2.]))
    },
    {
        "w_ii": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.]])),
        "w_hi": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.], [6., 7., 8.]])),
        "b_i": nn.parameter.Parameter(torch.tensor([11., 12, 13.])),

        "w_if": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.]])),
        "w_hf": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.], [6., 7., 8.]])),
        "b_f": nn.parameter.Parameter(torch.tensor([11., 12, 13.])),

        "w_ig": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.]])),
        "w_hg": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.], [6., 7., 8.]])),
        "b_g": nn.parameter.Parameter(torch.tensor([11., 12, 13.])),

        "w_io": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.]])),
        "w_ho": nn.parameter.Parameter(torch.tensor([[6., 7., 8.], [9., 10., 11.], [6., 7., 8.]])),
        "b_o": nn.parameter.Parameter(torch.tensor([11., 12, 13.])),

        "weight_sum_1": nn.parameter.Parameter(torch.tensor([3.])),
        "weight_sum_2": nn.parameter.Parameter(torch.tensor([4.]))
    },
    {
        "w_ii": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.]])),
        "w_hi": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.], [6., 7., 8.]])),
        "b_i": nn.parameter.Parameter(torch.tensor([-1., 4., 5.])),

        "w_if": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.]])),
        "w_hf": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.], [6., 7., 8.]])),
        "b_f": nn.parameter.Parameter(torch.tensor([-1., 4., 5.])),

        "w_ig": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.]])),
        "w_hg": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.], [6., 7., 8.]])),
        "b_g": nn.parameter.Parameter(torch.tensor([-1., 4., 5.])),

        "w_io": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.]])),
        "w_ho": nn.parameter.Parameter(torch.tensor([[0., -1., 2.], [3., -4., 5.], [6., 7., 8.]])),
        "b_o": nn.parameter.Parameter(torch.tensor([-1., 4., 5.])),

        "weight_sum_1": nn.parameter.Parameter(torch.tensor([5.])),
        "weight_sum_2": nn.parameter.Parameter(torch.tensor([6.]))
    },
    {
        "w_ii": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.]])),
        "w_hi": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.], [6., 7., 8.]])),
        "b_i": nn.parameter.Parameter(torch.tensor([0., 1., -1.])),

        "w_if": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.]])),
        "w_hf": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.], [6., 7., 8.]])),
        "b_f": nn.parameter.Parameter(torch.tensor([0., 1., -1.])),

        "w_ig": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.]])),
        "w_hg": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.], [6., 7., 8.]])),
        "b_g": nn.parameter.Parameter(torch.tensor([0., 1., -1.])),

        "w_io": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.]])),
        "w_ho": nn.parameter.Parameter(torch.tensor([[0., 1., -2.], [3., 4.,- 5.], [6., 7., 8.]])),
        "b_o": nn.parameter.Parameter(torch.tensor([0., 1., -1.])),

        "weight_sum_1": nn.parameter.Parameter(torch.tensor([7.])),
        "weight_sum_2": nn.parameter.Parameter(torch.tensor([8.]))
    }
]


def initialize_my_weights(cell: MDLSTMCell, direction: int):
    direction_weights = weights[direction]
    cell.w_ii = direction_weights["w_ii"]
    cell.w_hi = direction_weights["w_hi"]
    cell.b_i = direction_weights["b_i"]

    cell.w_if = direction_weights["w_if"]
    cell.w_hf = direction_weights["w_hf"]
    cell.b_f = direction_weights["b_f"]

    cell.w_ig = direction_weights["w_ig"]
    cell.w_hg = direction_weights["w_hg"]
    cell.b_g = direction_weights["b_g"]

    cell.w_io = direction_weights["w_io"]
    cell.w_ho = direction_weights["w_ho"]
    cell.b_o = direction_weights["b_o"]

    cell.weight_sum_1 = direction_weights["weight_sum_1"]
    cell.weight_sum_2 = direction_weights["weight_sum_2"]

torch.manual_seed(0)

NB_IMAGES = 10
images = torch.rand(NB_IMAGES, 2, 5, 4)

mdlstm_test = MDLSTM(height=5, width=4, in_channels=2, out_channels=3)
start = time.time()
computed = mdlstm_test.forward(images)
end = time.time()
print(f"It took {math.ceil((end - start) * 1000)}ms")
assert computed.shape == torch.Size([NB_IMAGES, 4, 3, 5, 4]), f"Computed output does not have the good shape ({computed.shape})"

############ First direction
# Compute and check first value
cell_value = images[:, :, 0, 0]
i = torch.sigmoid(cell_value.matmul(mdlstm_test.lstm_lr_tb.w_ii) + mdlstm_test.lstm_lr_tb.b_i)
f = torch.sigmoid(cell_value.matmul(mdlstm_test.lstm_lr_tb.w_if) + mdlstm_test.lstm_lr_tb.b_f)
g = torch.tanh(cell_value.matmul(mdlstm_test.lstm_lr_tb.w_ig) + mdlstm_test.lstm_lr_tb.b_g)
o = torch.sigmoid(cell_value.matmul(mdlstm_test.lstm_lr_tb.w_io) + mdlstm_test.lstm_lr_tb.b_o)
c00 = i * g
h00 = o * torch.tanh(c00)
c0 = c00 * (mdlstm_test.lstm_lr_tb.weight_sum_1 + mdlstm_test.lstm_lr_tb.weight_sum_2)
h0 = h00 * (mdlstm_test.lstm_lr_tb.weight_sum_1 + mdlstm_test.lstm_lr_tb.weight_sum_2)
assert torch.all(torch.abs(computed[:, 0, :, 0, 0] - h0) < 1e-7), f'First value in the first direction is not good : {computed[:, 0, :, 0, 0]} VS {h0}'
# Compute and check value (1,0)
cell_value = images[:, :, 1, 0]
i = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_ii + mdlstm_test.lstm_lr_tb.b_i + h0 @ mdlstm_test.lstm_lr_tb.w_hi)
f = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_if + mdlstm_test.lstm_lr_tb.b_f + h0 @ mdlstm_test.lstm_lr_tb.w_hf)
g = torch.tanh(cell_value @ mdlstm_test.lstm_lr_tb.w_ig + mdlstm_test.lstm_lr_tb.b_g + h0 @ mdlstm_test.lstm_lr_tb.w_hg)
o = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_io + mdlstm_test.lstm_lr_tb.b_o + h0 @ mdlstm_test.lstm_lr_tb.w_ho)
c10_0 = f * c0 + i * g
h10_0 = o * torch.tanh(c10_0)
i = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_ii + mdlstm_test.lstm_lr_tb.b_i)
f = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_if + mdlstm_test.lstm_lr_tb.b_f)
g = torch.tanh(cell_value @ mdlstm_test.lstm_lr_tb.w_ig + mdlstm_test.lstm_lr_tb.b_g)
o = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_io + mdlstm_test.lstm_lr_tb.b_o)
c10_1 = i * g
h10_1 = o * torch.tanh(c10_1)
c10 = c10_0 * mdlstm_test.lstm_lr_tb.weight_sum_1 + c10_1 * mdlstm_test.lstm_lr_tb.weight_sum_2
h10 = h10_0 * mdlstm_test.lstm_lr_tb.weight_sum_1 + h10_1 * mdlstm_test.lstm_lr_tb.weight_sum_2
assert torch.all(torch.abs(computed[:, 0, :, 1, 0] - h10) < 1e-7), f'First value in the first direction is not good : {computed[:, 0, :, 1, 0]} VS {h10}'

# Compute and check value (0,1)
cell_value = images[:, :, 0, 1]
i = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_ii + mdlstm_test.lstm_lr_tb.b_i + h0 @ mdlstm_test.lstm_lr_tb.w_hi)
f = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_if + mdlstm_test.lstm_lr_tb.b_f + h0 @ mdlstm_test.lstm_lr_tb.w_hf)
g = torch.tanh(cell_value @ mdlstm_test.lstm_lr_tb.w_ig + mdlstm_test.lstm_lr_tb.b_g + h0 @ mdlstm_test.lstm_lr_tb.w_hg)
o = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_io + mdlstm_test.lstm_lr_tb.b_o + h0 @ mdlstm_test.lstm_lr_tb.w_ho)
c01_0 = f * c0 + i * g
h01_0 = o * torch.tanh(c01_0)
i = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_ii + mdlstm_test.lstm_lr_tb.b_i)
f = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_if + mdlstm_test.lstm_lr_tb.b_f)
g = torch.tanh(cell_value @ mdlstm_test.lstm_lr_tb.w_ig + mdlstm_test.lstm_lr_tb.b_g)
o = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_io + mdlstm_test.lstm_lr_tb.b_o)
c01_1 = i * g
h01_1 = o * torch.tanh(c01_1)
c01 = c01_0 * mdlstm_test.lstm_lr_tb.weight_sum_2 + c01_1 * mdlstm_test.lstm_lr_tb.weight_sum_1
h01 = h01_0 * mdlstm_test.lstm_lr_tb.weight_sum_2 + h01_1 * mdlstm_test.lstm_lr_tb.weight_sum_1
assert torch.all(torch.abs(computed[:, 0, :, 0, 1] - h01) < 1e-7), f'First value in the first direction is not good : {computed[:, 0, :, 0, 1]} VS {h01}'
# Compute and check value (1,1)
cell_value = images[:, :, 1, 1]
i = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_ii + mdlstm_test.lstm_lr_tb.b_i + h01 @ mdlstm_test.lstm_lr_tb.w_hi)
f = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_if + mdlstm_test.lstm_lr_tb.b_f + h01 @ mdlstm_test.lstm_lr_tb.w_hf)
g = torch.tanh(cell_value @ mdlstm_test.lstm_lr_tb.w_ig + mdlstm_test.lstm_lr_tb.b_g + h01 @ mdlstm_test.lstm_lr_tb.w_hg)
o = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_io + mdlstm_test.lstm_lr_tb.b_o + h01 @ mdlstm_test.lstm_lr_tb.w_ho)
c11_0 = f * c01 + i * g
h11_0 = o * torch.tanh(c11_0)
i = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_ii + mdlstm_test.lstm_lr_tb.b_i + h10 @ mdlstm_test.lstm_lr_tb.w_hi)
f = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_if + mdlstm_test.lstm_lr_tb.b_f + h10 @ mdlstm_test.lstm_lr_tb.w_hf)
g = torch.tanh(cell_value @ mdlstm_test.lstm_lr_tb.w_ig + mdlstm_test.lstm_lr_tb.b_g + h10 @ mdlstm_test.lstm_lr_tb.w_hg)
o = torch.sigmoid(cell_value @ mdlstm_test.lstm_lr_tb.w_io + mdlstm_test.lstm_lr_tb.b_o + h10 @ mdlstm_test.lstm_lr_tb.w_ho)
c11_1 = f * c10 + i * g
h11_1 = o * torch.tanh(c11_1)
c11 = c11_0 * mdlstm_test.lstm_lr_tb.weight_sum_1 + c11_1 * mdlstm_test.lstm_lr_tb.weight_sum_2
h11 = h11_0 * mdlstm_test.lstm_lr_tb.weight_sum_1 + h11_1 * mdlstm_test.lstm_lr_tb.weight_sum_2
assert torch.all(torch.abs(computed[:, 0, :, 1, 1] - h11) < 1e-6), f'First value in the first direction is not good : {computed[:, 0, :, 1, 1]} VS {h11}'
