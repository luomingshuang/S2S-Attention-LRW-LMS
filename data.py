# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
import math
    
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, folds, vid_path, vid_pad, txt_pad):
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.folds = folds
        self.videos = glob.glob(os.path.join(vid_path, "*", "{}", "*").format(self.folds))
        # print(self.videos)
        self.videos = list(filter(lambda dir: len(os.listdir(dir)) == 29, self.videos))
        # print(self.videos)
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)
            # print(items, items[-2], items[-1])            
            self.data.append((vid, items[-3], items[-2]))        
        # print(self.data)
                
    def __getitem__(self, idx):
        (vid, anno_txt, name) = self.data[idx]
        vid = self._load_vid(vid)
        anno = self._load_anno(anno_txt)
        # anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        if(self.folds == 'train'):
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        vid = ColorNormalize(vid)
        # print('vid.transpose:', vid.transpose(3,0,1,2).shape)
        # print(anno_txt, anno)
        inputs = torch.FloatTensor(vid.transpose(3,0,1,2))
        labels = torch.LongTensor(anno)
        return {'encoder_tensor': inputs, 'decoder_tensor': labels}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        # files = sorted(os.listdir(p))
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (100, 50)) for im in array]
        array = np.stack(array, axis=0)
        return array
    
    def _load_anno(self, name):
        return MyDataset.txt2arr(name, 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
   
    @staticmethod
    def load_features(arr, filename, use_boundary=False, augment=True):
        if use_boundary:
            meta_path = filename.split('.')[0]+'.txt'
            with open(meta_path, 'r') as f:
                duration = math.ceil(float (f.readlines()[-1].split(' ')[1]) * 25)
                st = math.ceil((30 - duration)/2)
                ed = math.floor((30 + duration)/2)
        else: 
            st, ed = 0, 29
        return arr[st: ed]

    @staticmethod
    def txt2arr(txt, SOS=False):
        # SOS: 1, EOS: 2, P: 0, OTH: 3+x
        arr = []
        if(SOS):            
            tensor = [1]
        else:
            tensor = []
        for c in list(txt):
            tensor.append(3 + MyDataset.letters.index(c))
        tensor.append(2)
        return np.array(tensor)
    
    @staticmethod
    def tensor2text(tensor):
        # (B, T)
        result = []
        n = tensor.size(0)
        T = tensor.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = tensor[i,t]
                if(c == 2): break
                elif(c == 3): text.append(' ')
                elif(3 < c): text.append(chr(c-4+ord('a')))
            text = ''.join(text)
            result.append(text)
        return result

    @staticmethod
    def arr2txt(arr):       
        # (B, T)
        result = []
        n = arr.size(0)
        T = arr.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = arr[i,t]
                if(c >= 3):
                    text.append((MyDataset.letters[c - 3]).lower())
            text = ''.join(text)
            result.append(text)
        return result
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt)
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()        

    @staticmethod
    def ED(predict, truth):
        ED = [1.0*editdistance.eval(p[0], p[1]) for p in zip(predict, truth)] 
        return ED