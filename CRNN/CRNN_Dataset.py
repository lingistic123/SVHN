import os

import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import numpy as np
import os,glob
import random
from PIL import Image

train_path = 'data\\crnn_data\\train'
test_path = 'data\\crnn_data\\test'
valid_path = 'data\\crnn_data\\valid'

class SVHC_Dataset(Dataset):
    CHARS = '0123456789'
    char_to_seq = {char:i+1 for i,char in enumerate(CHARS)}
    seq_to_char = {label:char for char,label in char_to_seq.items()}
    
    
    
    def __init__(self,train_path,transform = None):
        self.transform = transform
        self.images_path = glob.glob(os.path.join(train_path,'*.png'))
        self.images_path.sort(key = lambda x:int(x.split('\\')[-1].split('_')[0]))
        self.channel = 1
        
    def __getitem__(self,index):
        img = Image.open(self.images_path[index]).convert('L')
        img = np.array(img)
        img = img.reshape((self.channel,32,32))# channel  height  width
        img = (img/127.0) - 1.0
        img = torch.FloatTensor(img)
        if self.transform is not None:
            img = self.transform(img)
        char = list(self.images_path[index].split('\\')[-1].split('_')[1])
        length = int(self.images_path[index].split('\\')[-1].split('_')[-1].split('.')[0])
        seq =[self.char_to_seq[i] for i in char]
        seq = torch.LongTensor(seq)
        return img,seq,torch.LongTensor([length])
    
    
    def __len__(self):
        return len(self.images_path)


def SVHC_collate_fn(batch):
    # zip(*batch)拆包
    images, targets, target_lengths = zip(*batch)
    # stack就是向量堆叠的意思。一定是扩张一个维度，然后在扩张的维度上，把多个张量纳入仅一个张量。想象向上摞面包片，摞的操作即是stack，0轴即按块stack
    images = torch.stack(images, 0)
    # cat是指向量拼接的意思。一定不扩张维度，想象把两个长条向量cat成一个更长的向量。
    targets = torch.cat(targets, 0) #将多个batch并到一个上面
    target_lengths = torch.cat(target_lengths, 0)
    # 此处返回的数据即使train_loader每次取到的数据，迭代train_loader，每次都会取到三个值，即此处返回值。
    return images, targets, target_lengths

