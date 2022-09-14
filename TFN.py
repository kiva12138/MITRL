import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, type_rnn, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.type_rnn = type_rnn
        rnn = nn.LSTM if type_rnn.lower() == 'lstm' else nn.GRU

        self.rnn = rnn(in_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        self.rnn.flatten_parameters()
        _, final_states = self.rnn(x)
        if self.type_rnn == 'lstm':
            final_states = final_states[0]  
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, input_dims, hidden_dims, out, dropouts, post_fusion_dim):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.out= out
        self.post_fusion_dim = post_fusion_dim

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.audio_subnet = TextSubNet(self.audio_in, 'gru', self.audio_hidden, self.out[0],  num_layers=2, bidirectional=True, dropout=self.audio_prob)
        self.video_subnet = TextSubNet(self.video_in, 'gru',self.video_hidden, self.out[1],  num_layers=2, bidirectional=True, dropout=self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, 'gru', self.text_hidden, self.out[2],  num_layers=2, bidirectional=True, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.out[0] + 1) * (self.out[1] + 1) * (self.out[2] + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)


    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)
            video_x: tensor of shape (batch_size, sequence_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)
        

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        # post_fusion_y_2 = post_fusion_y_1
        # post_fusion_y_3 = torch.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        post_fusion_y_3 = self.post_fusion_layer_3(post_fusion_y_2)
        # output = post_fusion_y_3 * self.output_range + self.output_shift
        output = post_fusion_y_3

        return output

if __name__ == '__main__':
    a = torch.randn(16, 20, 74)
    v = torch.randn(16, 20, 35)
    t = torch.randn(16, 20, 300)

    # ts = TextSubNet(300, 'gru', hidden_size=256, out_size=256, num_layers=2, dropout=0.2, bidirectional=True)
    # r = ts(t)
    # print(r.shape)

    # exit(0)

    tfn = TFN([74, 35, 300], [128, 128, 128], [128, 128, 128], [0.5, 0.5, 0.5, 0.5], 64)
    out = tfn(a, v, t)
    print(out.shape)
