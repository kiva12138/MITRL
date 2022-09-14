import os
import numpy as np
import pickle
import torch
import h5py
from torch.utils.data import DataLoader, Dataset

data_path = r'/mnt/data2/Sh/ShData/MM'

def regreto7class(a):
    # [-3,3] => 7-class
    if a < -2:
        res = -3
    if -2 <= a and a < -1:
        res = -2
    if -1 <= a and a < 0:
        res = -1
    if 0 <= a and a <= 0:
        res = 0
    if 0 < a and a <= 1:
        res = 1
    if 1 < a and a <= 2:
        res = 2
    if a > 2:
        res = 3
    return res + 3

def pomregreto7class(a):
    # [1,7] => 7-class
    if a < 2:
        res = -3
    if 2 <= a and a < 3:
        res = -2
    if 3 <= a and a < 4:
        res = -1
    if 4 <= a and a < 5:
        res = 0
    if 5 <= a and a < 6:
        res = 1
    if 6 <= a and a < 7:
        res = 2
    if a >= 7:
        res = 3
    return res + 3

def regreto2class(a):
    # [-3,3] => 2-class
    if a >= 0:
        res = 1
    else:
        res = 0
    return res


# Support Dataset:
# Youtube(3) MMMO(2) MOUD(2) POM(regr*17) IEMOCAP(9) IEMOCAP_20(4*2) MOSI_20(regr,2,7) MOSI_50(regr,2,7) MOSEI_20(regr,2,7) MOSEI_50(regr,2,7)
class UnifiedDataset(Dataset):
    def __init__(self, mode, normalize=False, type='MMMO'):
        assert mode in ['test', 'train', 'valid']
        assert type in ['Youtube', 'MMMO', 'MOUD', 'POM', 'IEMOCAP', 'IEMOCAP_20', 'MOSI_20', 'MOSI_50', 'MOSEI_20', 'MOSEI_50']

        if type in  ['Youtube', 'MMMO', 'MOUD', 'POM', 'IEMOCAP']:
            audio = np.load(os.path.join(data_path, type, 'covarep_'+mode+'.p'),mmap_mode='r', allow_pickle=True, encoding='bytes',)
            video = np.load(os.path.join(data_path, type, 'facet_'+mode+'.p'),mmap_mode='r', allow_pickle=True, encoding='bytes',)
            text  = np.load(os.path.join(data_path, type, 'text_'+mode+'.p'),mmap_mode='r', allow_pickle=True, encoding='bytes',)
            label = np.load(os.path.join(data_path, type, 'y_'+mode+'.p'),mmap_mode='r', allow_pickle=True, encoding='bytes',)
        elif type in ['IEMOCAP_20', 'MOSI_50', 'MOSEI_50']:
            with open(os.path.join(data_path, type, 'data.pkl'), "rb") as f:
                data = pickle.load(f)[mode]
            text = data['text']
            audio = data['audio']
            video = data['vision']
            label = data['labels']
        elif type in ['MOSI_20']:
            with h5py.File(os.path.join(data_path, type, 'X_'+mode+'.h5'), "r") as f:
                data = np.array(f['data'])    
                text = data[:,:, :300]
                audio = data[:,:, 300:305]
                video = data[:,:, 305:]
            with h5py.File(os.path.join(data_path, type, 'y_'+mode+'.h5'), "r") as f:
                label = np.array(f['data'])
        elif type in ['MOSEI_20']:
            with h5py.File(os.path.join(data_path, type, 'audio_'+mode+'.h5'), "r") as f:
                audio = f.get('d1')[:]
            with h5py.File(os.path.join(data_path, type, 'video_'+mode+'.h5'), "r") as f:
                video = f.get('d1')[:]
            with h5py.File(os.path.join(data_path, type, 'text_'+mode+'_emb.h5'), "r") as f:
                text = f.get('d1')[:]
            with h5py.File(os.path.join(data_path, type, 'y_'+mode+'.h5'), "r") as f:
                label = f.get('d1')[:]
        else:
            raise NotImplementedError
            
        if normalize:
            audio = audio/np.max(np.abs(audio))
            video = video/np.max(np.abs(video))
            text  = text/np.max(np.abs(text))

        self.type = type
        
        label_new = []
        for item in label:
            if type in ['Youtube', 'MOUD']:
                item_new = int(torch.argmax(torch.Tensor(item), dim=0).item())
            elif type in ['MMMO']:
                item_new = int(item > 3.5)
            elif type in ['POM']:
                # item_new = [float(item_) for item_ in item]
                item_new = [item[-3], pomregreto7class(item[-3])]
            elif type in ['IEMOCAP']:
                item_new = int(torch.argmax(torch.Tensor(list(item)), dim=0).item())
            elif type in ['IEMOCAP_20']:
                item_new = [int(torch.argmax(torch.Tensor(item_), dim=0).item()) for item_ in item]
                assert sum(item_new) == 1
                item_new = int(torch.argmax(torch.Tensor(item_new), dim=0).item())
            elif type in ['MOSI_20']:
                item_new = [float(item), regreto2class(item), regreto7class(item)]
            elif type in ['MOSI_50']:
                item_new = [float(item[0][0]), regreto2class(item[0][0]), regreto7class(item[0][0])]
            elif type in ['MOSEI_20']:
                item_new = [float(item), regreto2class(item), regreto7class(item)]
            elif type in ['MOSEI_50']:
                item_new = [float(item[0][0]), regreto2class(item[0][0]), regreto7class(item[0][0])]
            else:
                raise NotImplementedError
            label_new.append(item_new)
                
        # Clean IEMOCAP
        if type == 'IEMOCAP':
            new_audio, new_video, new_text, new_labels = [], [], [], []
            for audio_, video_, text_, label_ in zip(audio, video, text, label_new):
                if label_ == 0 or label_ == 10:
                    continue
                label_ -= 1
                assert label_ >=0 and label_ <= 9
                new_audio.append(audio_)
                new_video.append(video_)
                new_text.append(text_)
                new_labels.append(label_)
            audio, video, text, label_new = new_audio, new_video, new_text, new_labels

        self.audio, self.video, self.text, self.label = audio, video, text, label_new

    def __getitem__(self, index):
        return self.audio[index], self.video[index], self.text[index], self.label[index]

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    
    import faulthandler
    faulthandler.enable()
    dataset = UnifiedDataset(mode='test', normalize=False, type='POM')
    data_loader = DataLoader(dataset, 8)
    print('All samples:', len(dataset))
    for _, data in enumerate(data_loader):
        print([data_.shape for data_ in data[:-1]])
        print(data[-1])
