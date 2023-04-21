import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def get_predictions_tensor(predictions):
    pred_vals, pred_indices = torch.max(predictions, dim=-1)
    return pred_indices


# 0.5 0.5 norm
def showImageNormalized(data):
    data = data.numpy().transpose((1, 2, 0))
    data = data / 2 + 0.5
    plt.imshow(data)
    plt.show()


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-1 * logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class SimilarityLossAVT(torch.nn.Module):
    def __init__(self, result_loss, loss_similarity, gamma=0.5):
        super(SimilarityLossAVT, self).__init__()
        self.result_loss, self.loss_similarity, self.gamma  = result_loss, loss_similarity, gamma

        if loss_similarity == 'KL':
            self.similarity_loss = F.kl_div
        elif loss_similarity == 'Cosine':
            self.similarity_loss = nn.CosineSimilarity(dim=-1)
        else:
            raise NotImplementedError

    def forward(self, inputs, target):
        prediction, V_F, A_F, T_F = inputs

        loss_result = self.result_loss(prediction, target)

        # Similarity measured by KL
        if self.loss_similarity == 'KL':
            loss_similarityv_a = 0.5 * (self.similarity_loss(V_F, A_F, reduction='batchmean') + self.similarity_loss(A_F, V_F, reduction='batchmean'))
            loss_similarityv_t = 0.5 * (self.similarity_loss(V_F, T_F, reduction='batchmean') + self.similarity_loss(T_F, V_F, reduction='batchmean'))
            loss_similaritya_t = 0.5 * (self.similarity_loss(T_F, A_F, reduction='batchmean') + self.similarity_loss(A_F, T_F, reduction='batchmean'))
            loss_similarityv_a = 0 if torch.isnan(loss_similarityv_a) else loss_similarityv_a
            loss_similarityv_t = 0 if torch.isnan(loss_similarityv_t) else loss_similarityv_t
            loss_similaritya_t = 0 if torch.isnan(loss_similaritya_t) else loss_similaritya_t

            loss_similarity = (loss_similaritya_t + loss_similarityv_a + loss_similarityv_t) / 3.0
        elif self.loss_similarity == 'Cosine':
            loss_similarityv_a = 1.0 - self.similarity_loss(V_F, A_F).mean()
            loss_similarityv_t = 1.0 - self.similarity_loss(V_F, T_F).mean()
            loss_similaritya_t = 1.0 - self.similarity_loss(A_F, T_F).mean()
            loss_similarity = (loss_similaritya_t + loss_similarityv_a + loss_similarityv_t) / 3.0
        else:
            raise NotImplementedError

        loss_all = self.gamma * loss_similarity + (1-self.gamma) * loss_result
        return loss_all

# TODO: for gau, var can be calculate by meaned feature or single
class SMLossAVT(torch.nn.Module):
    def __init__(self, sm_loss, sm_level):
        super(SMLossAVT, self).__init__()
        self.sm_loss, self.sm_level = sm_loss, sm_level

        if sm_loss.lower() == 'kl':
            self.loss_func = MultVariateKLD(reduction='mean')
        elif sm_loss.lower() == 'gau':
            self.loss_func = nn.GaussianNLLLoss(reduction='mean', full=False)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        V_F, A_F, T_F = inputs
        assert len(V_F.shape) == 3 and len(A_F.shape) == 3 and len(T_F.shape) == 3, \
            'Features should have shape(bs, len, d) not '+str(V_F.shape)

        if self.sm_loss == 'kl':
            var_a, var_v, var_t = A_F.var(dim=-1), V_F.var(dim=-1), T_F.var(dim=-1)
            a_, v_, t_ = A_F.mean(-1), V_F.mean(-1), T_F.mean(-1)
            
            output_av = (self.loss_func(a_, v_, var_a, var_v) + self.loss_func(v_, a_, var_v, var_a))/2
            output_at = (self.loss_func(a_, t_, var_a, var_t) + self.loss_func(t_, a_, var_t, var_a))/2
            output_vt = (self.loss_func(v_, t_, var_v, var_t) + self.loss_func(t_, v_, var_t, var_v))/2

            loss_all = (output_at + output_av + output_vt)/3
        elif self.sm_loss == 'gau':
            sumed = (V_F + A_F + T_F) / 3
            if self.sm_level == 'batch':
                var = sumed.mean(-1).mean(-1).var(dim=-1)

                output_a = self.loss_func(A_F.mean(-1).mean(-1), sumed.mean(-1).mean(-1), var)
                output_v = self.loss_func(V_F.mean(-1).mean(-1), sumed.mean(-1).mean(-1), var)
                output_t = self.loss_func(T_F.mean(-1).mean(-1), sumed.mean(-1).mean(-1), var)

                loss_all = (output_a + output_v + output_t)/3
            elif self.sm_level == 'time':
                var = sumed.mean(-1).var(dim=-1)

                output_a = self.loss_func(A_F.mean(-1), sumed.mean(-1), var)
                output_v = self.loss_func(V_F.mean(-1), sumed.mean(-1), var)
                output_t = self.loss_func(T_F.mean(-1), sumed.mean(-1), var)

                loss_all = (output_a + output_v + output_t)/3
                # print(var.detach().cpu())
                # print(torch.log(var))
                # print(output_a.detach().cpu().item(), output_v.detach().cpu().item(),
                #     output_t.detach().cpu().item(), loss_all.detach().cpu().item())
            elif self.sm_level == 'vec':
                var = sumed.var(dim=-1)

                output_a = self.loss_func(A_F, sumed, var)
                output_v = self.loss_func(V_F, sumed, var)
                output_t = self.loss_func(T_F, sumed, var)

                loss_all = (output_a + output_v + output_t)/3
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return loss_all


class UnivariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(UnivariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, var_1, var_2): # var is variance
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = var_1.type(dtype=torch.float64)  # sigma_1 := sigma_nat^2
        sigma_2 = var_2.type(dtype=torch.float64)  # sigma_2 := sigma_adv^2

        # log(sqrt(sigma2)/sqrt(sigma1))
        term_1 = (sigma_2.sqrt() / sigma_1.sqrt()).log()

        # (sigma_1 + (mu1-mu2)^2)/(2*sigma_2)
        term_2 = (sigma_1 + (mu1 - mu2).pow(2))/(2*sigma_2)

        # Calc kl divergence on entire batch
        kl = term_1 + term_2 - 0.5

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')

        return kl_agg

        
class MultVariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(MultVariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, var_1, var_2): # var is standard deviation
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = var_1.type(dtype=torch.float64)
        sigma_2 = var_2.type(dtype=torch.float64)

        sigma_diag_1 = torch.diag_embed(sigma_1, offset=0, dim1=-2, dim2=-1)
        sigma_diag_2 = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1)

        sigma_diag_2_inv = sigma_diag_2.inverse()

        # log(det(sigma2^T)/det(sigma1))
        term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()
        # term_1[term_1.ne(term_1)] = 0

        # trace(inv(sigma2)*sigma1)
        term_2 = torch.diagonal((torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3 = torch.matmul(torch.matmul((mu2 - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_2_inv),
                              (mu2 - mu1).unsqueeze(-1)).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n = mu1.shape[1]

        # Calc kl divergence on entire batch
        kl = 0.5 * (term_1 - n + term_2 + term_3)

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')

        return kl_agg


def make_weights_for_balanced_classes(labels, nclasses=7):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        if count[i] == 0:
            continue
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]

    return weight


def make_weights_for_balanced_classes_from_list(labels, nclasses=7):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in labels:
        count[int(item)] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[int(val)]

    return weight


def getResNet18Extractor(pretrained=False):
    import torchvision
    resnet18 = torchvision.models.resnet18(pretrained=pretrained)
    conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    a_modules = [conv0] + list(resnet18.children())[1:-1]
    resnet18 = torch.nn.Sequential(*a_modules)
    return resnet18


def getResNet50Extractor(pretrained=False):
    import torchvision
    resnet18 = torchvision.models.resnet50(pretrained=pretrained)
    conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    a_modules = [conv0] + list(resnet18.children())[1:-1]
    resnet18 = torch.nn.Sequential(*a_modules)
    return resnet18


def getAlexNetExtractor(pretrained=False):
    import torchvision
    alexnet = torchvision.models.alexnet(pretrained=pretrained)
    conv0 = torch.nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    average_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
    a_modules = [conv0] + list(list(alexnet.children())[0][1:]) + [average_pool]
    alexnet = torch.nn.Sequential(*a_modules)
    return alexnet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).contiguous()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1).contiguous()
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe.shape)
        # print(x.shape)
        # print(self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, feature_dim=512, nhead=1, nhid=2048, nlayers=6, dropout=0.5, max_seq_len=512):
        super(TransformerModel, self).__init__()

        self.pos_encoder = PositionalEncoding(feature_dim, dropout, max_seq_len)
        encoder_layer = TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        norm = nn.LayerNorm(normalized_shape=feature_dim, eps=1e-6)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=nlayers, norm=norm)

    # x[batch, n, dim]
    def forward(self, x, mask):
        x = x.permute(1, 0, 2).contiguous()
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2).contiguous()
        return x


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollateForSequence:
    def __init__(self, dim=0, pad_tensor_pos=[2,3], data_kind=4):
        self.dim = dim
        self.pad_tensor_pos = pad_tensor_pos
        self.data_kind = data_kind

    def pad_collate(self, batch):
        new_batch = []

        for pos in range(self.data_kind):
            if pos not in self.pad_tensor_pos:
                if not isinstance(batch[0][pos], torch.Tensor):
                    new_batch.append(torch.Tensor([x[pos] for x in batch]))
                else:
                    new_batch.append(torch.stack([x[pos] for x in batch]), dim=0)
            else:
                max_len = max(map(lambda x: x[pos].shape[self.dim], batch))
                padded = list(map(lambda x: pad_tensor(x[pos], pad=max_len, dim=self.dim), batch))
                padded = torch.stack(padded, dim=0)
                new_batch.append(padded)

        return new_batch

    def __call__(self, batch):
        return self.pad_collate(batch)


def get_mask_from_sequence(sequence, dim):
    return (torch.sum(torch.abs(sequence), dim=dim) == 0)


def load_general_model(model, path_):
    old_state_dict = torch.load(path_)['model']
    new_state_dict = model.state_dict()
    for name in old_state_dict:
        new_name = name.replace('module.', '')
        if 'output.1.' in new_name:
            continue
        new_state_dict[new_name] = old_state_dict[name]
    model.load_state_dict(new_state_dict)
    return model


def lock_all_params(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False
    return model


if __name__ == '__main__':
    loss = SimilarityLossAVT(torch.nn.CrossEntropyLoss(), 'KL', 0.5)
    # loss = SimilarityLossAVT(torch.nn.CrossEntropyLoss(), 'Cosine', 0.5)
    predictions, target = torch.randn((8, 3)), torch.randint(0, 3, (8, ))
    V_F, A_F, T_F = torch.randn((8, 128)), torch.randn((8, 128)), torch.randn((8, 128))
    final_loss = loss((predictions, V_F, A_F, T_F), target)
    print(final_loss)
