import torch
import torch.nn as nn
from NaiveTransformerExpr.EncoderMultiStream import Encoder as TransformerEncoder


def aug_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    std_features = torch.std(data, dim=aug_dim)
    max_features, _ = torch.max(data, dim=aug_dim)
    min_features, _ = torch.min(data, dim=aug_dim)
    union_feature = torch.cat((mean_features, std_features, min_features, max_features), dim=-1)
    return union_feature

def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features

class ModelSimilarity(nn.Module):
    def __init__(self, d_a, d_v, d_t, d_inner, layers, n_head, dropout, d_out, feature_aug, feature_compose, add_sa, num_class, n_position=30):
        super(ModelSimilarity, self).__init__()
        self.feature_aug, self.feature_compose = feature_aug, feature_compose

        self.tf_encoder_av = TransformerEncoder(d_a, d_v, layers, d_inner, n_head, d_k=None, d_v=None, dropout=dropout, n_position=n_position, add_sa=add_sa)
        self.tf_encoder_at = TransformerEncoder(d_a, d_t, layers, d_inner, n_head, d_k=None, d_v=None, dropout=dropout, n_position=n_position, add_sa=add_sa)
        self.tf_encoder_vt = TransformerEncoder(d_v, d_t, layers, d_inner, n_head, d_k=None, d_v=None, dropout=dropout, n_position=n_position, add_sa=add_sa)

        # Conv1D
        self.conv_a = nn.Conv1d(in_channels=d_a, out_channels=d_a, kernel_size=3, stride=1, padding=1, groups=n_head)
        self.conv_v = nn.Conv1d(in_channels=d_v, out_channels=d_v, kernel_size=3, stride=1, padding=1, groups=n_head)
        self.conv_t = nn.Conv1d(in_channels=d_t, out_channels=d_t, kernel_size=3, stride=1, padding=1, groups=n_head)

        # BN + ReLU
        self.bn_a = nn.BatchNorm1d(d_a)
        self.relu_a = nn.ReLU(inplace=True)
        self.bn_v = nn.BatchNorm1d(d_v)
        self.relu_v = nn.ReLU(inplace=True)
        self.bn_t = nn.BatchNorm1d(d_t)
        self.relu_t = nn.ReLU(inplace=True)

        self.dropout_a = nn.Dropout(dropout)
        self.dropout_v = nn.Dropout(dropout)
        self.dropout_t = nn.Dropout(dropout)
        
        # FC
        self.fc_a = nn.Linear(d_a, d_out)
        self.fc_v = nn.Linear(d_v, d_out)
        self.fc_t = nn.Linear(d_t, d_out)

        # Classifer
        if self.feature_aug == 'aug':
            fc_dimension = d_out * 4
        elif self.feature_aug == 'mean':
            fc_dimension = d_out
        else:
            raise NotImplementedError
        if self.feature_compose == 'cat':
            fc_dimension = fc_dimension * 3
        elif self.feature_compose == 'mean' or self.feature_compose == 'sum':
            fc_dimension = fc_dimension
        else:
            raise NotImplementedError
        if d_out <= 128:
            self.classifier = nn.Sequential(
                nn.Linear(fc_dimension, num_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fc_dimension, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_class),
            )

    # V: (bs, len1, dim1)
    # A: (bs, len2, dim2)
    # T: (bs, len3, dim3)
    def forward(self, A, V,  T, src_maska=None, src_maskv=None, src_maskt=None, return_features=False):
        # Temporal
        A, V, T = self.conv_a(A.transpose(1, 2)), self.conv_v(V.transpose(1, 2)), self.conv_t(T.transpose(1, 2))
        A, V, T = self.relu_a(self.bn_a(A)).transpose(1, 2), self.relu_v(self.bn_v(V)).transpose(1, 2), self.relu_t(self.bn_t(T)).transpose(1, 2)
        A, V, T = self.dropout_a(A), self.dropout_v(V), self.dropout_t(T)

        A_, V_, T_ = A, V, T
        A_1, V_1 = self.tf_encoder_av(A_, V_, src_maska, src_maskv)
        A_2, T_1 = self.tf_encoder_at(A_, T_, src_maska, src_maskt)
        V_2, T_2 = self.tf_encoder_vt(V_, T_, src_maskv, src_maskt)
        # A, V, T = A_1+A_2, V_1+V_2, T_1+T_2
        A, V, T = (A_1+A_2)/2.0, (V_1+V_2)/2.0, (T_1+T_2)/2.0

        # Dimensional
        A_F_ = self.fc_a(A)
        V_F_ = self.fc_v(V)
        T_F_ = self.fc_t(T)
        
        # Out
        if self.feature_aug == 'aug':
            A_F, V_F, T_F = aug_temporal(A_F_, 1), aug_temporal(V_F_, 1), aug_temporal(T_F_, 1)
        elif self.feature_aug == 'mean':
            A_F, V_F, T_F = mean_temporal(A_F_, 1), mean_temporal(V_F_, 1), mean_temporal(T_F_, 1)
        else:
            raise NotImplementedError
        
        if self.feature_compose == 'sum':
            features = torch.stack([A_F, V_F, T_F], dim=-1).sum(dim=-1)
        elif self.feature_compose == 'cat':
            features = torch.cat([A_F, V_F, T_F], dim=-1)
        elif self.feature_compose == 'mean':
            features = torch.stack([A_F, V_F, T_F], dim=-1).mean(dim=-1)
        else:
            raise NotImplementedError

        result = self.classifier(features)
        # result = torch.sigmoid(self.classifier(features))

        if return_features:
            return result, V_F_, A_F_, T_F_
        else:
            return result


if __name__ == '__main__':
    model = ModelSimilarity(d_a=74, d_v=35, d_t=300, d_inner=512, layers=2, n_head=4, dropout=0.5, d_out=64, feature_aug='mean', feature_compose='sum',num_class=1, add_sa=True)
    v = torch.randn(8, 21, 35)
    a = torch.randn(8, 21, 74)
    t = torch.randn(8, 21, 300)
    result = model(a, v, t)
    print(result.shape)