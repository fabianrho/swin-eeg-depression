import torch
# from torchvision import transforms, datasets
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image
import torchvision
import os
import glob
# import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt



class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, resize_to = (None,None), three_channels = False, raw = True, file_ids = None):
        super(EEGDataset, self).__init__()

        self.resize_to = resize_to
        self.three_channels = three_channels
        self.raw = raw

        if raw:
            self.files = glob.glob(os.path.join(folder_path,'*.npy'))
        else:
            self.files = glob.glob(os.path.join(folder_path,'*.png'))

        if file_ids is not None:
            self.files = [file for file in self.files if file.split("/")[-1].split("_")[0] in file_ids]

        # self.files = self.files[:1000]



            

    def __getitem__(self, index):
        data_path = self.files[index]

        # label = [1,0] if data_path.split('/')[-1].split('_')[0][0] == "H" else [0,1]
        label = 0 if data_path.split('/')[-1].split('_')[0][0] == "H" else 1

        # print(img_path.split('/')[-1].split('_')[0].split('.')[0])


        if self.raw:
            data = torch.Tensor(np.load(data_path)).unsqueeze(0)
        else:   

            data = read_image(data_path)[0,:,:].unsqueeze(0)/255

        if self.resize_to != (None,None) and not self.raw:
            data = torchvision.transforms.Resize(self.resize_to)(torch.tensor(data))
        elif self.resize_to != (None,None) and self.raw:
            data = torchvision.transforms.Resize(self.resize_to)(data)

        # create three channels
        if self.three_channels:
            data = torch.cat([data,data,data], dim=0)

        # normalize data
        # data = (data - data.mean()) / data.std()

        # normalize between 0 and 1
        data = (data - data.min()) / (data.max() - data.min())

        if label == 0:
            label = [1,0]
        else:
            label = [0,1]


        return data, torch.tensor(label).float()
    
    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    dataset = EEGDataset('data_images')

    # print(len(dataset))
    # for i in range(len(dataset)):
    for i in range(10):

        data = dataset[i]
        print(data)