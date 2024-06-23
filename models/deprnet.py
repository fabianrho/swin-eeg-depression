import torch
import torch.nn as nn
import sys
sys.path.append('.')

class DeprNet(nn.Module):
    def __init__(self):
        super(DeprNet, self).__init__()
        self.C1 = nn.Conv2d(1,128,kernel_size=(1,5))
        self.N1 = nn.BatchNorm2d(128)
        self.M1 = nn.MaxPool2d(kernel_size=(1,2))

        self.C2 = nn.Conv2d(128,64,kernel_size=(1,5))
        self.N2 = nn.BatchNorm2d(64)
        self.M2 = nn.MaxPool2d(kernel_size=(1,2))

        self.C3 = nn.Conv2d(64,64,kernel_size=(1,5))
        self.N3 = nn.BatchNorm2d(64)
        self.M3 = nn.MaxPool2d(kernel_size=(1,2))

        self.C4 = nn.Conv2d(64,32,kernel_size=(1,3))
        self.N4 = nn.BatchNorm2d(32)
        self.M4 = nn.MaxPool2d(kernel_size=(1,2))

        self.C5 = nn.Conv2d(32,32,kernel_size=(1,2))
        self.N5 = nn.BatchNorm2d(32)
        self.M5 = nn.MaxPool2d(kernel_size=(1,2))

        self.D1 = nn.LazyLinear(16)
        self.D2 = nn.Linear(16, 8)
        self.D3 = nn.Linear(8, 2)

        self.relu = nn.LeakyReLU()


        
    def forward(self, x):
        x = self.C1(x)
        x = self.N1(x)
        x = nn.LeakyReLU()(x)
        x = self.M1(x)

        x = self.C2(x)
        x = self.N2(x)
        x = nn.LeakyReLU()(x)

        x = self.M2(x)

        x = self.C3(x)
        x = self.N3(x)
        x = nn.LeakyReLU()(x)
        x = self.M3(x)

        x = self.C4(x)
        x = self.N4(x)
        x = nn.LeakyReLU()(x)
        x = self.M4(x)

        x = self.C5(x)
        x = self.N5(x)
        x = nn.LeakyReLU()(x)
        x = self.M5(x)

        x = x.view(x.size(0), -1)

        x = self.D1(x)
        x = nn.LeakyReLU()(x)
        x = self.D2(x)
        x = nn.LeakyReLU()(x)
        x = self.D3(x)
        x = nn.Softmax(dim=1)(x)


        return x



if __name__ ==  "__main__":
    from data_loader import EEGDataset
    import torch
    from transformers import AutoImageProcessor, Swinv2ForImageClassification, ResNetForImageClassification, DeiTForImageClassification
    import transformers
    import tqdm

    import matplotlib.pyplot as plt


    # device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")



    dataset = EEGDataset("data/data_4s_0.75overlap", resize_to=(None,None), raw = True, three_channels=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=False, drop_last=True)

    model = DeprNet()

    pass