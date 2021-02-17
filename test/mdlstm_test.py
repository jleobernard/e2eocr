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

image_1 = torch.Tensor([
    # First channel
    [[0,1,2,3],
     [4,5,6,7],
     [8,9,10,11],
     [12,13,14,15],
     [16,17,18,19]]
    ,
    # Second channel
    [[-10, -11, -12, -13],
     [-14, -15, -16, -17],
     [-18, -19, -110, -111],
     [-112, -113, -114, -115],
     [-116, -117, -118, -119]]
])
mdlstm_test = MDLSTM(height=5, width=4, in_channels=2, out_channels=3)
initialize_my_weights(mdlstm_test.lstm_lr_tb, 0)
initialize_my_weights(mdlstm_test.lstm_rl_tb, 1)
initialize_my_weights(mdlstm_test.lstm_lr_bt, 2)
initialize_my_weights(mdlstm_test.lstm_rl_bt, 3)
start = time.time()
computed = mdlstm_test.forward(image_1.unsqueeze(0))
end = time.time()
print(f"It took {math.ceil((end - start) * 1000)}ms")
assert computed.shape == torch.Size([1, 4, 3, 5, 4]), 'Computed output does not have the good shape'
v0 = torch.sigmoid(torch.tensor([-21., -30., -39.]))
c_0_0 = v0.mul(torch.tanh(torch.tensor([-21., -30., -39.])))
h_0_0 = v0.mul(torch.tanh(c_0_0)) * 3.
assert torch.all(computed[0, 0, :, 0, 0].eq(h_0_0)), f'First value in the first direction is not good : {computed[0, 0, :, 0, 0]} VS {h_0_0}'
#assert torch.all(computed[0, 0, :, 0, 1].eq(weights[0]["bias"])), 'First value in the first direction is not good'
#assert torch.all(computed[0, 0, :, 1, 0].eq(weights[0]["bias"])), 'First value in the first direction is not good'
#assert torch.all(computed[0, 0, :, 1, 1].eq(weights[0]["bias"])), 'First value in the first direction is not good'
