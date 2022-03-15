import sys
sys.path.append('./')
import argparse
import pdb
import argparse
import yaml 
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from tqdm import tqdm
#from train import Trainer
#from test import total_test
#from setup import setup_solver

import torch
import torchaudio
import torchaudio.functional
import logging
from glob import glob
data_path = '/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/*/*'
#/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/s1/
music_list = glob(data_path)
data_path2 = '/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/test/*/*'
music_list2 = glob(data_path2)
#pdb.set_trace()
print("music list :", len(music_list), len(music_list2))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 기본적으로 44.1kHz, => Win_len = 40ms, n_fft = win_len보다 크게. hop_len = 10ms.. 굳이?
win_len = 2048
hop_length = 512
n_fft = 2048
fs = 4
duration_len = 3
save_path = '/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max_spectrogram/'

frame_num = 128 
audio_maxlen = int(frame_num*256*16-1) 
window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
best_loss = 10
sampling = 16000
length = sampling * duration_len
source_list = ['bass.wav', 'drums.wav', 'mixture.wav', 'other.wav', 'vocals.wav']
pkl_list = ['bass.pkl', 'drums.pkl', 'mixture.pkl', 'other.pkl', 'vocals.pkl']
i= 0 
for music_name in music_list:
    music_n = music_name.split('/')[-1]
    dir_n = music_name.split('/')[-2]
    music_n = music_n.split('.wav')[0]
    vocal_signal, vocal_fs = torchaudio.load(music_name) # 1, 235200
    vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False, return_complex = True)#, return_complex = False)
    #vocal_spectrogram = 1 X 1025 X 460
    vocal_real_0 = vocal_spectrogram[0, :256, :]
    
    '''
    input_real_0 = input_spectrogram[0,:,:,0] # B, 2, 1025, 259
    input_imag_0 = input_spectrogram[0,:,:,1] # B, 2, 1025, 259
    input_real_1 = input_spectrogram[1,:,:,0] # B, 2, 1025, 259?
    input_imag_1 = input_spectrogram[1,:,:,1] # B, 2, 1025, 259
    '''
    vocal_real_0 = vocal_real_0.numpy()
    print(vocal_real_0.shape)
    out_dir = save_path + 'train-360/' + dir_n + '/' + music_n + '.npy'
    #print("outdir =", out_dir)
    np.save(out_dir, vocal_real_0)
    if i % 1000 ==0 :
        print("Done %i/%i"%(i, len(music_list)))
    i+=1
for music_name in music_list2:
    music_n = music_name.split('/')[-1]
    dir_n = music_name.split('/')[-2]
    music_n = music_n.split('.wav')[0]
    vocal_signal, vocal_fs = torchaudio.load(music_name) # 1, 235200
    vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False,return_complex = True)#, return_complex = False)
    #vocal_spectrogram = 1 X 1025 X 460
    vocal_real_0 = vocal_spectrogram[0, :256, :]
    
    '''
    input_real_0 = input_spectrogram[0,:,:,0] # B, 2, 1025, 259
    input_imag_0 = input_spectrogram[0,:,:,1] # B, 2, 1025, 259
    input_real_1 = input_spectrogram[1,:,:,0] # B, 2, 1025, 259?
    input_imag_1 = input_spectrogram[1,:,:,1] # B, 2, 1025, 259
    '''
    vocal_real_0 = vocal_real_0.numpy()
    out_dir = save_path + 'test/' + dir_n + '/' + music_n + '.npy'
    print("outdir =", out_dir)
    #pdb.set_trace()
    #os.makedirs(out_dir , exist_ok = True)
    np.save(out_dir, vocal_real_0)
    #with open(out_dir + '/spectrogram.pkl', 'wb') as f:
    #    pickle.dump(batch_dict, f)
        