import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    """
    From https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros((max_len, max_len, d_model))
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class AttentionHead(nn.Module):

    def __init__(self, in_channel: int = 512, q_k_v_channel: int = 64):
        super(AttentionHead, self).__init__()
        self.q_k_v_channel = q_k_v_channel
        self.weights = nn.parameter.Parameter(torch.empty((in_channel, q_k_v_channel * 3)))
        self.score_norm = math.sqrt(q_k_v_channel)

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        weigthed_x = x @ self.weights
        q, k, v = weigthed_x.chunk(3, 1)
        attn = nn.functional.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.score_norm, dim=-1)
        return torch.matmul(attn, v), attn


class SpatialAttention(nn.Module):

    def __init__(self, in_channel: int = 512, q_k_v_channel: int = 64, out_channel: int = 512,
                 nb_att_heads: int = 8, needs_pos_encoding: bool = False):
        super(SpatialAttention, self).__init__()
        if needs_pos_encoding:
            self.pe = PositionalEncoding(in_channel, 0)
        else:
            self.pe = None
        self.heads = nn.ModuleList([AttentionHead(in_channel, q_k_v_channel) for _ in range(nb_att_heads)])
        self.w = nn.Linear(q_k_v_channel * nb_att_heads, out_channel)

    def initialize_weights(self):
        for head in self.heads:
            head.initialize_weights()

    def forward(self, x):
        """
        :param x: is a list of image so the shape is (batch, in_channel, H, W)
        :return: (batch, out_channel, H, W)
        """
        _, _, height, width = x.shape
        if self.pe:
            x = self.pe(x)
        # Flatten the image to get a sequence-like representation of the image
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2) # (batch, H x W, in_channel) = x.shape
        computeds = [head(x) for head in self.heads]
        cated = torch.cat(computeds, dim=-1)  # (batch, H x W, nb_head x head_channel)
        x = self.w(cated)  # (batch, H x W, out_channel)
        # Now we need to make it look like the original image again
        x = nn.functional.fold(x, output_size=(height, width), kernel_size=(1, 1))
        return x
