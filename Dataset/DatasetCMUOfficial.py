import torch
import os
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Shapes of features: [(8, 300), (8,), (8, 300), (8, 768), (8, 768), (8, 47), (8, 35), (8, 430), (8, 74), (8, 1585), (8, 384)]
# Features' name: [_words, _words_text, emb2, emb_bert_hidden, emb_bert_pool, _visua47, _visua35, _visua430, _acoustic74, _acoustic1585, _acoustic384]
# feature, label, label2, label7, segment
MOSI_PATH = '/mnt/data2/Sh/ShData/cmumosi/'

# Shapes of features: [(55, 300), (55,), (55, 300), (55, 768), (55, 768), (55, 35), (55, 713), (55, 74)]
# Features' name:  [_words, _words_text, emb2,emb_bert_hidden, emb_bert_pool, _visua35,_visua713, _acoustic74,]
# feature, label, label2, label7, segment
MOSEI_PATH = '/mnt/data2/Sh/ShData/cmumosei/'

def multi_collate(batch):
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.Tensor([sample[1] for sample in batch]).reshape(-1,).float()
    labels_2 = torch.Tensor([sample[2] for sample in batch]).reshape(-1,).long()
    labels_7 = torch.Tensor([sample[3] for sample in batch]).reshape(-1,).long()
    sentences = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], padding_value=0).transpose(0, 1)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], padding_value=0).transpose(0, 1)
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], padding_value=0).transpose(0, 1)
    
    return sentences, visual, acoustic, labels, labels_2, labels_7# , lengths

def multi_collate_repeat(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.Tensor([sample[1] for sample in batch]).reshape(-1,)
    labels_2 = torch.Tensor([sample[2] for sample in batch]).reshape(-1,).long()
    labels_7 = torch.Tensor([sample[3] for sample in batch]).reshape(-1,).long()

    max_len = max([sample[0][0].shape[0] for sample in batch])
    sentences, visual, acoustic = [], [], []
    for sample in batch:
        sentence_i, visual_i, acoustic_i = sample[0][0], sample[0][1], sample[0][2]
        while sentence_i.shape[0] < max_len:
            sentence_i = np.concatenate([sentence_i, sentence_i], axis=0)
        sentence_i = sentence_i[:max_len]
        sentences.append(sentence_i)
        while visual_i.shape[0] < max_len:
            visual_i = np.concatenate([visual_i, visual_i], axis=0)
        visual_i = visual_i[:max_len]
        visual.append(visual_i)
        while acoustic_i.shape[0] < max_len:
            acoustic_i = np.concatenate([acoustic_i, acoustic_i], axis=0)
        acoustic_i = acoustic_i[:max_len]
        acoustic.append(acoustic_i)

    sentences = torch.FloatTensor(sentences)
    visual = torch.FloatTensor(visual)
    acoustic = torch.FloatTensor(acoustic)
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, labels_2, labels_7, lengths

def regression2classification(a):
    if a < -2:
        res = 0
    elif a < -1:
        res = 1
    elif a < 0:
        res = 2
    elif a <= 0:
        res = 3
    elif a <= 1:
        res = 4
    elif a <= 2:
        res = 5
    elif a > 2:
        res = 6
    else:
        print(a)
        return 3
    return res

def regression2classification_bin(a):
    if a < 0:
        res = 0
    else:
        res = 1
    return res

# Text: 300-1, 300-2, 768-1, 768-2,
# Audio: 74, 1585, 384
# Video: 47, 35, 430, 
def get_mosi_dataset(save_path, text='300-1', audio='74', video='47'):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    train, dev, test = data[0], data[1], data[2]
    train = [list([list(train_i[0]), train_i[1], train_i[2], train_i[3],train_i[4], ]) for train_i in train] 
    dev = [list([list(dev_i[0]), dev_i[1], dev_i[2], dev_i[3],dev_i[4], ]) for dev_i in dev] 
    test = [list([list(test_i[0]), test_i[1], test_i[2], test_i[3],test_i[4], ]) for test_i in test]

    if audio == '74':
        for train_i in train:
            del  train_i[0][10]
            del  train_i[0][9]
        for dev_i in dev:
            del  dev_i[0][10] 
            del dev_i[0][9]
        for test_i in test:
            del  test_i[0][10] 
            del test_i[0][9]
    elif audio == '1585':
        for train_i in train:
            del  train_i[0][10] 
            del  train_i[0][8]
        for dev_i in dev:
            del  dev_i[0][10] 
            del  dev_i[0][8]
        for test_i in test:
            del  test_i[0][10] 
            del test_i[0][8]
    elif audio == '384':
        for train_i in train:
            del  train_i[0][9]
            del  train_i[0][8] 
        for dev_i in dev:
            del dev_i[0][9]
            del  dev_i[0][8] 
        for test_i in test:
            del test_i[0][9]
            del  test_i[0][8] 
    else:
        raise NotImplementedError

    if video == '47':
        for train_i in train:
            del  train_i[0][7]
            del train_i[0][6]
        for dev_i in dev:
            del  dev_i[0][7] 
            del dev_i[0][6]
        for test_i in test:
            del  test_i[0][7] 
            del test_i[0][6]
    elif video == '35':
        for train_i in train:
            del  train_i[0][7]
            del train_i[0][5]
        for dev_i in dev:
            del  dev_i[0][7] 
            del dev_i[0][5]
        for test_i in test:
            del  test_i[0][7] 
            del test_i[0][5]
    elif video == '430':
        for train_i in train:
            del train_i[0][6]
            del  train_i[0][5] 
        for dev_i in dev:
            del dev_i[0][6]
            del  dev_i[0][5] 
        for test_i in test:
            del  test_i[0][6]
            del  test_i[0][5] 
    else:
        raise NotImplementedError
        
    if text == '300-1':
        for train_i in train:
            del  train_i[0][4] 
            del  train_i[0][3]
            del  train_i[0][2]
            del train_i[0][1]
        for dev_i in dev:
            del  dev_i[0][4] 
            del  dev_i[0][3]
            del  dev_i[0][2]
            del dev_i[0][1]
        for test_i in test:
            del  test_i[0][4] 
            del  test_i[0][3]
            del  test_i[0][2]
            del test_i[0][1]
    elif text == '300-2':
        for train_i in train:
            del  train_i[0][4] 
            del  train_i[0][3]
            del  train_i[0][1]
            del  train_i[0][0]
        for dev_i in dev:
            del  dev_i[0][4] 
            del  dev_i[0][3]
            del  dev_i[0][1]
            del  dev_i[0][0]
        for test_i in test:
            del  test_i[0][4] 
            del  test_i[0][3]
            del  test_i[0][1]
            del  test_i[0][0]
    elif text == '768-1':
        for train_i in train:
            del  train_i[0][4] 
            del  train_i[0][2]
            del  train_i[0][1]
            del  train_i[0][0]
        for dev_i in dev:
            del  dev_i[0][4] 
            del  dev_i[0][2]
            del  dev_i[0][1]
            del  dev_i[0][0]
        for test_i in test:
            del  test_i[0][4] 
            del  test_i[0][2]
            del  test_i[0][1]
            del  test_i[0][0]
    elif text == '768-2':
        for train_i in train:
            del  train_i[0][3]
            del  train_i[0][2] 
            del  train_i[0][1]
            del  train_i[0][0]
        for dev_i in dev:
            del  dev_i[0][3]
            del  dev_i[0][2] 
            del  dev_i[0][1]
            del  dev_i[0][0]
        for test_i in test:
            del  test_i[0][3]
            del  test_i[0][2] 
            del  test_i[0][1]
            del  test_i[0][0]
    else:
        raise NotImplementedError    
    
    return train, dev, test


# Text: 300-1, 300-2, 768-1, 768-2,
# Audio: 74
# Video: 35, 713
def get_mosei_dataset(save_path, text='300-1', audio='74', video='47'):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    train, dev, test = data[0], data[1], data[2]
    train = [list([list(train_i[0]), train_i[1], train_i[2], train_i[3],train_i[4], ]) for train_i in train] 
    dev = [list([list(dev_i[0]), dev_i[1], dev_i[2], dev_i[3],dev_i[4], ]) for dev_i in dev] 
    test = [list([list(test_i[0]), test_i[1], test_i[2], test_i[3],test_i[4], ]) for test_i in test]

    if audio == '74':
        pass
    else:
        raise NotImplementedError

    if video == '35':
        for train_i in train:
            del train_i[0][6]
        for dev_i in dev:
            del dev_i[0][6]
        for test_i in test:
            del test_i[0][6]
    elif video == '713':
        for train_i in train:
            del train_i[0][5]
        for dev_i in dev:
            del dev_i[0][5]
        for test_i in test:
            del test_i[0][5] 
    else:
        raise NotImplementedError
        
    if text == '300-1':
        for train_i in train:
            del  train_i[0][4] 
            del  train_i[0][3]
            del  train_i[0][2]
            del train_i[0][1]
        for dev_i in dev:
            del  dev_i[0][4] 
            del  dev_i[0][3]
            del  dev_i[0][2]
            del dev_i[0][1]
        for test_i in test:
            del  test_i[0][4] 
            del  test_i[0][3]
            del  test_i[0][2]
            del test_i[0][1]
    elif text == '300-2':
        for train_i in train:
            del  train_i[0][4] 
            del  train_i[0][3]
            del train_i[0][1]
            del  train_i[0][0]
        for dev_i in dev:
            del  dev_i[0][4] 
            del  dev_i[0][3]
            del dev_i[0][1]
            del  dev_i[0][0]
        for test_i in test:
            del  test_i[0][4] 
            del  test_i[0][3]
            del test_i[0][1]
            del  test_i[0][0]
    elif text == '768-1':
        for train_i in train:
            del  train_i[0][4] 
            del  train_i[0][2]
            del train_i[0][1]
            del  train_i[0][0]
        for dev_i in dev:
            del  dev_i[0][4] 
            del  dev_i[0][2]
            del dev_i[0][1]
            del  dev_i[0][0]
        for test_i in test:
            del  test_i[0][4] 
            del  test_i[0][2]
            del test_i[0][1]
            del  test_i[0][0]
    elif text == '768-2':
        for train_i in train:
            del  train_i[0][3]
            del  train_i[0][2] 
            del train_i[0][1]
            del  train_i[0][0]
        for dev_i in dev:
            del  dev_i[0][3]
            del dev_i[0][1]
            del  dev_i[0][2] 
            del  dev_i[0][0]
        for test_i in test:
            del  test_i[0][3]
            del  test_i[0][2] 
            del test_i[0][1]
            del  test_i[0][0]
    else:
        raise NotImplementedError    
    
    return train, dev, test

if __name__ == '__main__':
    train, valid, test = get_mosi_dataset(os.path.join(MOSI_PATH, 'mosi_filter_0norm.pkl'), text='300-2', audio='384', video='35')
    data_loader = DataLoader(train, 8, collate_fn=multi_collate)
    print('All samples:', len(train))
    for _, data in enumerate(data_loader):
        print([data_.shape for data_ in data])
        print(data[3])
        print(data[4])
        print(data[5])
    del train, valid, test

    train, valid, test = get_mosei_dataset(os.path.join(MOSEI_PATH, 'mosei_filter_0norm.pkl'), text='768-2', audio='74', video='35')
    data_loader = DataLoader(test, 8, collate_fn=multi_collate)
    print('All samples:', len(train))
    for _, data in enumerate(data_loader):
        print([data_.shape for data_ in data])
        print(data[3])
        print(data[4])
        print(data[5])
    del train, valid, test