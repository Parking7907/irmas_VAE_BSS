from calendar import c
from email.errors import ObsoleteHeaderDefect
from socketserver import DatagramRequestHandler
import sys
sys.path.append('..')
import torch

from torch.utils.data.dataset import Dataset
#from torchvision.transforms import Normalize
from pathlib import Path
#import soundfile as sf
import pickle
import pdb
#import cv2
import numpy as np
import torchvision
import os
import random
from glob import glob
#from augmentation import augmentation
import torchaudio
import torch.nn.functional as F
# import albumentations as A
def get_data_loaders(data_dir,batch_size,data_size,data_len,kwargs):
    #data_dir = '/home/data/jinyoung/source_separation/IRMAS/spectrogram/'
    train_set = IRMAS(data_dir, data_partition = 'train', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = data_size, data_len = data_len)
    test_set = IRMAS(data_dir, data_partition = 'test', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = data_size, data_len = data_len)
    #pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


class IRMAS(Dataset):
    def __init__(self, data_dir, data_partition = 'train', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = 256, data_len =32):
        super(IRMAS, self).__init__()
        #data_path = /home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/s1,s2,mix_clean,mix_both,mix_single
        self.data_dir = data_dir
        self.data_partition = data_partition    
        self.duration_len = time_duration
        self.sample_rate = sample_rate
        self.sources_bass = []
        self.sources_drum = []
        self.sources_mixture = []
        self.sources_others = []
        self.sources_vocal = []
        self.data_size = data_size
        self.data_len = data_len
        #if data_partition == 'train':
        #    data_path = data_dir + 'train-360'
        #else:
        #    data_path = data_dir + 'test'
        data_path = data_dir + "*/*"
        #source1_data_path =data_path + "/s1/*/"
        #source2_data_path = data_path + "/s2/*/"
        #print(data_path)
        self.music_list = glob(data_path)
        self.music_list.sort()
        self.seed = seed
        self.target = target
        self.random_mix = random_mix
        #print(self.music_list)
        i = 0
        #Seed 고정.. Required?
        #random.seed(self.seed)
        #Read data path
            
    def __len__(self):
        return len(self.music_list) * 10

    def __getitem__(self, idx):
        idx = idx % len(self.music_list)
        data_path = self.music_list[idx]
        #"id": music_n, "vocal": vocal_real_0, "bass":bass_real_0,"drums":drums_real_0, "other":other_real_0, "mixture":mixture_real_0}
        source_list = ['s1', 's2']
        #source_list = ['vocal', 'bass', 'drums', 'other']
        data_p = self.music_list[idx].split('/')
        music_n = data_p[-1] #'5007-31603-0016_2688-144987-0062.npy'
        inst_list = ['cel','cla','flu', 'gac', 'gel','org','pia','sax','tru','vio','voi']
        inst_n = data_p[-2]
        #print(inst_list, inst_n)
        inst_list.remove(inst_n)
        #print(music_n, inst_n)
        inst_selected=random.sample(inst_list,1)
        
        s2_path = self.data_dir + inst_selected[0] + '/*'
        #print(s2_path)
        s2_list = glob(s2_path)
        s2_num = random.randint(0, len(s2_list)-1)
        s2_p = s2_list[s2_num]
        dir_n = '/'.join(data_p[:-2]) #'/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max_spectrogram/train-360/
        
        #s1_path = dir_n + '/s1/' + music_n
        #s2_path = dir_n + '/s2/' + music_n
        #print(data_path, s1_path, s2_path)
        source_1 = np.load(data_path)
        source_2 = np.load(s2_p)
        data = source_1 + source_2
        #print(data.shape, source_1.shape, source_2.shape)
        
        if data.ndim == 3:
            data_real = data[:,:,0] + data[:,:,1] * 1j
            data = np.absolute(data_real)
        else:
            data = np.absolute(data)
        if source_1.ndim ==3:
            source_1_real = source_1[:,:,0] + source_1[:,:,1] * 1j
            source_1 = np.absolute(source_1_real)
        else:
            source_1 = np.absolute(source_1)
        if source_2.ndim ==3:
            source_2_real = source_2[:,:,0] + source_2[:,:,1] * 1j
            source_2 = np.absolute(source_2_real)
        else:
            source_2 = np.absolute(source_2)
        
        #print(data.shape, source_1.shape, source_2.shape)

        #Output size 문제 때문에....
        data = data[:self.data_size, :self.data_len]
        source_1 = source_1[:self.data_size, :self.data_len]
        source_2 = source_2[:self.data_size, :self.data_len]

        #print(data.shape, source_1.shape, source_2.shape)
        data = torch.Tensor(data)
        source_1 = torch.Tensor(source_1)
        source_2 = torch.Tensor(source_2)
        ## For Padding
        if len(data[0]) - self.data_len -1 < 0 :
            start = 0
            #print(data.shape, source_1.shape, source_2.shape)
            pad_to = (0,self.data_len - len(data[0]))
            data = F.pad(data, pad_to, "constant", 0)
            source_1 = F.pad(source_1, pad_to, "constant", 0)
            source_2 = F.pad(source_2, pad_to, "constant", 0)
            #print(data.shape, source_1.shape, source_2.shape)
        else:
            start = random.randint(0, len(data[0]) - self.data_len -1)
            source_1 = source_1[:, start:start + self.data_len]
            source_2 = source_2[:, start:start + self.data_len]
            data = data[:, start:start + self.data_len]
        
        #start = 0

        
        source_1 = self.normalize(source_1)
        source_2 = self.normalize(source_2)
        source_out = self.normalize(data)
        #print(source_out.shape, source_1.shape, source_2.shape)
        #print(torch.max(source_1), torch.max(source_2), torch.max(source_out), torch.min(source_1), torch.min(source_2), torch.min(source_out))

        return source_1, source_2, source_out


    def normalize(self, data):
        #print("Norm")
        

        data = data.float()
        #mean = torch.mean(data.float())
        #std = torch.std(data.float())
        #data = (data - mean) / std
        max_ = torch.max(data)
        min_ = torch.min(data)
        if max_ == 0 or min_ == 0:
            pass
            #print("NULL!!!!!!!!!!!!!!!!!!!!!!!!..", max_, min_)
        else:
            if max_ > (-1 * min_):
                data = data / (max_)
            else:
                data = data / (-1 * min_)
        #print(max_, min_, torch.max(data), torch.min(data), vocal_idx, bass_idx, drum_idx, others_idx)
        return data


print("new")

data_dir = '/home/data/jinyoung/source_separation/IRMAS/spectrogram/'
batch_size = 16
kwargs = {'num_workers': 32, 'pin_memory': True}
data_size = 256
data_len = 32
train_loader, test_loader = get_data_loaders(data_dir, batch_size, data_size, data_len, kwargs)
#pdb.set_trace()
print("done")

