import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask1=None, mask2=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask1 is not None and mask2 is not None:
            # print('Attention', attn.shape, 'Mask', mask1.shape)
            # print('Attention', attn.shape, 'Mask', mask2.shape)
            # # print(mask)
            attn = attn.masked_fill_(mask1, 1e-9) # Fills elements of att with 1e-9 where mask is True.
            attn = attn.masked_fill_(mask2, 1e-9) # Fills elements of att with 1e-9 where mask is True.
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


if __name__ == '__main__':
    batch_size, n_head = 16, 8
    n_hid_k, n_hid_v = 128, 200
    seq_len_q, seq_len_k = 10, 15
    q_ = torch.randn(batch_size, n_head, seq_len_q, n_hid_k)
    k_ = torch.randn(batch_size, n_head, seq_len_k, n_hid_k)
    v_ = torch.randn(batch_size, n_head, seq_len_k, n_hid_v)

    scaledDotProductAttention = ScaledDotProductAttention(temperature=1)
    result, attention = scaledDotProductAttention(q_, k_, v_)
    print(result.shape, attention.shape)