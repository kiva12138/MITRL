import torch.nn as nn
import torch
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, d_hid, n_position=100):
        super(PositionEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(int(pos_i)) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # [1,N,d]

    def forward(self, x):
        # x [B,N,d]
        # print(x.shape ,self.pos_table[:, :x.size(1)].shape)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


if __name__ == '__main__':
    batch_size, seq_len, n_hid = 16, 10, 128
    x_ = torch.zeros(batch_size, seq_len, n_hid)
    pe = PositionEncoding(d_hid=n_hid, n_position=seq_len)
    y_ = pe(x_)
    print(y_.shape)
