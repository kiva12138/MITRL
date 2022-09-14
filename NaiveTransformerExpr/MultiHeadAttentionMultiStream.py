import torch
import torch.nn as nn
import torch.nn.functional as F
from TFN import TFN

from torch.nn.parameter import Parameter
from NaiveTransformerExpr.ScaledDotProductAttention import ScaledDotProductAttention
# from ScaledDotProductAttention import ScaledDotProductAttention

class FBP(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, fbp_hid, fbp_k, dropout):
        super(FBP, self).__init__()
        self.fusion_1_matrix = nn.Linear(d_emb_1, fbp_hid*fbp_k, bias=False)
        self.fusion_2_matrix = nn.Linear(d_emb_2, fbp_hid*fbp_k, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_pooling = nn.AvgPool1d(kernel_size=fbp_k)
        self.fbp_k = fbp_k

    def forward(self, seq1, seq2):
        seq1 = self.fusion_1_matrix(seq1)
        seq2 = self.fusion_2_matrix(seq2)
        fused_feature = torch.mul(seq1, seq2)
        if len(fused_feature.shape) == 2:
            fused_feature = fused_feature.unsqueeze(0)
        fused_feature = self.fusion_dropout(fused_feature)
        fused_feature = self.fusion_pooling(fused_feature).squeeze(0) * self.fbp_k # (bs, 512)
        fused_feature = F.normalize(fused_feature, dim=-1, p=2)
        return fused_feature

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_emb_q, d_emb_v, d_k=None, d_v=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_emb_q, self.d_emb_v, self.n_head = d_emb_q, d_emb_v, n_head
        self.d_k = d_k if d_k is not None else d_emb_q
        self.d_v = d_v if d_v is not None else d_emb_v

        assert self.d_k % n_head == 0, 'Error from MultiHeadAttention: self.d_k % n_head should be zero.'
        assert self.d_v % n_head == 0, 'Error from MultiHeadAttention: self.d_v % n_head should be zero.'
        
        self.w_q = nn.Linear(d_emb_q, self.d_k, bias=False)
        self.w_k = nn.Linear(d_emb_v, self.d_k, bias=False)
        self.w_v = nn.Linear(d_emb_v, self.d_v, bias=False)
        self.fc = nn.Linear(self.d_v, d_emb_q, bias=False)

        self.fbp = FBP(self.d_k, self.d_k, 32, 2, dropout)
        self.fc_gate = nn.Linear(32, 1)
        self.gate_activate = nn.Tanh()
        # self.gate_activate = nn.Sigmoid()
        # self.tfn = TFN(
        #     input_dims=[self.d_k, self.d_k, self.d_v], 
        #     hidden_dims=[64, 64, 64], out=[64, 64, 64], 
        #     dropouts=[dropout, dropout, dropout, dropout], 
        #     # dropouts=[0, 0, 0, 0], 
        #     post_fusion_dim=32)
            
        # self.output_thredshole = Parameter(torch.FloatTensor([0]), requires_grad=True)

        # self.w_q = nn.Conv1d(d_emb_q, self.d_k, 3, 1, 1, bias=False)
        # self.w_k = nn.Conv1d(d_emb_v, self.d_k, 3, 1, 1, bias=False)
        # self.w_v = nn.Conv1d(d_emb_v, self.d_v, 3, 1, 1, bias=False)
        # self.fc = nn.Conv1d(self.d_v, d_emb_q, 3, 1, 1, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_emb_q, eps=1e-6)

    def forward(self, q, k, v, mask1=None, mask2=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        assert len_k == len_v, 'len_k should be equal with len_v.'

        residual = q
        # print(q.shape)

        # Separate different heads: b x l x n x (d/n)
        # q = self.w_q(q.transpose(1, 2)).transpose(1, 2).view(sz_b, len_q, n_head, d_k // n_head)
        # k = self.w_k(k.transpose(1, 2)).transpose(1, 2).view(sz_b, len_k, n_head, d_k // n_head)
        # v = self.w_v(v.transpose(1, 2)).transpose(1, 2).view(sz_b, len_v, n_head, d_v // n_head)
        q = self.w_q(q).view(sz_b, len_q, n_head, d_k // n_head)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k // n_head)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v // n_head)

        # q_, k_ = q.view(sz_b, len_q, d_k).mean(1), k.view(sz_b, len_k, d_k).mean(1)
        # q_, k_, v_ = q.view(sz_b, len_q, d_k), k.view(sz_b, len_k, d_k), v.view(sz_b, len_v, d_v)
        q_, k_ = q.view(sz_b, len_q, d_k), k.view(sz_b, len_k, d_k)
        gate_ = self.fbp(q_, k_)
        # gate_ = self.tfn(q_, k_, v_)
        gate_ = self.gate_activate(self.fc_gate(gate_))#.double()
        # gate_ = self.gate_activate(gate_).double()
        # gate_ = torch.where(gate_ > 0.0, gate_, 0.0).double()
        # gate_ = torch.where(gate_ <=0.0, gate_, 1.0).float()
        # gate_ = torch.where(gate_ > 0.0, 1.0, 0.0).float()
        gate_sign = gate_ / torch.abs(gate_)
        # print(gate_sign.detach().cpu().numpy().tolist())
        gate_ = (gate_sign + torch.abs(gate_sign)) / 2.0
        # print(gate_.detach().cpu().numpy().tolist())
        # print(gate_.requires_grad, gate_.grad_fn, end= ' \n' )

        # print(gate_.requires_grad, gate_.grad_fn, gate_tmp.requires_grad, gate_tmp.grad_fn, end= '\n' )
        # print((gate_>0).float().detach().sum().cpu().numpy() / len(gate_), end=' ')

        # Transpose for attention dot product: b x n x l x d
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(1).unsqueeze(-1)  # For head axis broadcasting.
            mask2 = mask2.unsqueeze(1).unsqueeze(2)  # For head axis broadcasting.

        # result b x n x lq x (dv/n)
        result, attn = self.attention(q, k, v, mask1=mask1, mask2=mask2)

        # Transpose to move the head dimension back: b x l x n x (dv/n)
        # Combine the last two dimensions to concatenate all the heads together: b x l x (dv)
        result = result.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        # b x l x (d_model)
        # result = self.dropout(self.fc(result.transpose(1, 2)).transpose(1, 2))
        result = self.dropout(self.fc(result))
        if len(gate_.shape) == 2:
            gate_ = gate_.unsqueeze(-1)
        result = result * gate_
        
        # print(gate_.requires_grad, end= '' )
        # result = result.masked_fill(gate_ < 0, 0)
        result += residual

        result = self.layer_norm(result)
        result = result.masked_fill(torch.isnan(result), 0.0)

        return result, attn


if __name__ == '__main__':
    q_ = torch.randn(8, 20, 256)
    k_ = torch.randn(8, 10, 128)
    v_ = torch.randn(8, 10, 128)

    mha = MultiHeadAttention(n_head=4, d_emb_q=256, d_emb_v=128, d_k=512, d_v=1024)
    result, attn = mha(q_, k_, v_)

    print(result.shape)
