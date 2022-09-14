from torch import nn
import torch

from NaiveTransformerExpr.EncodingLayerMultiStream import EncoderLayer as EncoderLayerNoSA
from NaiveTransformerExpr.EncodingLayerMultiStreamWithSA import EncoderLayer as EncoderLayerAddSA
from NaiveTransformerExpr.PositionEncoding import PositionEncoding
# from EncodingLayerMultiStream import EncoderLayer as EncoderLayerNoSA
# from EncodingLayerMultiStreamWithSA import EncoderLayer as EncoderLayerAddSA
# from PositionEncoding import PositionEncoding

def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features

class Encoder(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, n_layers, d_inner, n_head, d_k=None, d_v=None, dropout=0.1, n_position=2048, add_sa=False):
        super(Encoder, self).__init__()

        self.position_enc1 = PositionEncoding(d_emb_1, n_position=n_position)
        self.position_enc2 = PositionEncoding(d_emb_2, n_position=n_position)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_emb_1, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_emb_2, eps=1e-6)

        if not add_sa:
            self.layer_stack1 = nn.ModuleList([
                EncoderLayerNoSA(n_head, d_emb_1, d_emb_2, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)])
            self.layer_stack2 = nn.ModuleList([
                EncoderLayerNoSA(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)])
        else:
            self.layer_stack1 = nn.ModuleList([
                EncoderLayerAddSA(n_head, d_emb_1, d_emb_2, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)])
            self.layer_stack2 = nn.ModuleList([
                EncoderLayerAddSA(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)])


    def forward(self, seq1, seq2, src_mask1=None, src_mask2=None, return_attns=False):
        enc_slf_attn_list1, enc_slf_attn_list2 = [], []

        enc_output1 = self.layer_norm1(self.dropout1(self.position_enc1(seq1)))
        enc_output2 = self.layer_norm2(self.dropout2(self.position_enc2(seq2)))

        enc_output1 = enc_output1.masked_fill(torch.isnan(enc_output1), 0.0)
        enc_output2 = enc_output2.masked_fill(torch.isnan(enc_output2), 0.0)

        for enc_layer1, enc_layer2 in zip(self.layer_stack1, self.layer_stack2):
            temp_enc1, temp_enc2 = enc_output1, enc_output2
            enc_output1, enc_slf_attn1 = enc_layer1(temp_enc1, temp_enc2, temp_enc2, slf_attn_mask1=src_mask1, slf_attn_mask2=src_mask2)
            enc_output2, enc_slf_attn2 = enc_layer2(temp_enc2, temp_enc1, temp_enc1, slf_attn_mask1=src_mask2, slf_attn_mask2=src_mask1)
            enc_slf_attn_list1 += [enc_slf_attn1] if return_attns else []
            enc_slf_attn_list2 += [enc_slf_attn2] if return_attns else []

        if return_attns:
            return enc_output1, enc_output2, enc_slf_attn_list1, enc_slf_attn_list2
        return enc_output1, enc_output2


def make_mask(feature):
    return torch.sum(torch.abs(feature), dim=-1) == 0


if __name__ == '__main__':
    encoder = Encoder(d_emb_1=128, d_emb_2=256, n_layers=1, d_inner=512, n_head=2)
    a = torch.randn(4, 4, 128)
    b = torch.randn(4, 6, 256)
    a_mask, b_mask = make_mask(a), make_mask(b)
    a_mask = torch.Tensor([
        [False, False, False, False],
        [False, False, False, True],
        [False, False, False, True],
        [False, True, True, True],
    ]).long().bool()
    b_mask = torch.Tensor([
        [False, False, False, False, False, False,],
        [False, False, False, False, False, True],
        [False, False, False, False, False, True],
        [False, False, False, True, True, True],
    ]).long().bool()
    y= encoder(a, b, src_mask1=None, src_mask2=None, return_attns=False)
    print([y_.shape for y_ in y])
