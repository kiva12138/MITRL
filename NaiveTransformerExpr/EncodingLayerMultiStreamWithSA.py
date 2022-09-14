from torch import nn

from NaiveTransformerExpr.MultiHeadAttentionMultiStream import MultiHeadAttention
from NaiveTransformerExpr.PositionwiseFeedForward import PositionwiseFeedForward
# from MultiHeadAttentionMultiStream import MultiHeadAttention
# from PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, n_head, d_emb_q, d_emb_v, d_inner, d_k=None, d_v=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_emb_q=d_emb_q, d_emb_v=d_emb_v, d_k=d_k, d_v=d_v, dropout=dropout)
        self.slf_attn_sa = MultiHeadAttention(n_head=n_head, d_emb_q=d_emb_q, d_emb_v=d_emb_q, d_k=d_k, d_v=d_k, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_emb_q, d_inner, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask1=None, slf_attn_mask2=None):
        enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask1=slf_attn_mask1, mask2=slf_attn_mask2)
        enc_output, enc_slf_attn = self.slf_attn_sa(q, q, q, mask1=slf_attn_mask1, mask2=slf_attn_mask2)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
