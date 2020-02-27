import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):

    def __init__(self, window_size, n_channels=3, n_out=3):
        super().__init__()
        self.in_conv = FirstConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.fc0 = SeqToImage(window_size, 1)
        self.fc1 = SeqToImage(window_size, 1)
        self.fc2 = SeqToImage(window_size, 1)
        self.fc3 = SeqToImage(window_size, 1)
        self.fc4 = SeqToImage(window_size, 1)
        
        self.out_conv = LastConv(64, n_out)

    def forward(self, x):
#         print('Entrada:')
#         print(x.shape)
        x1 = self.in_conv(x)
#         print('In:')
#         print(x1.shape)
        x2 = self.down1(x1)
#         print('Down 1:')
#         print(x2.shape)
        x3 = self.down2(x2)
#         print('Down 2:')
#         print(x3.shape)
        x4 = self.down3(x3)
#         print('Down 3:')
#         print(x4.shape)
        x5 = self.down4(x4)
#         print('Down 4')
#         print(x5.shape)

        # Transformaçao de 3D para 2D
#         print('---------')
#         print(x5.shape)
        x = self.fc0(x5)
#         print('---------')
#         print(x.shape)

        x = self.up1(x, self.fc1(x4))
#         print('Up 1')
#         print(x.shape)
        x = self.up2(x, self.fc2(x3))
#         print('Up 2')
#         print(x.shape)
        x = self.up3(x, self.fc3(x2))
#         print('Up 3')
#         print(x.shape)
        x = self.up4(x, self.fc4(x1))
#         print('Up 4')
#         print(x.shape)
        x = self.out_conv(x)
#         print('Out')
#         print(x.shape)

        return x

#Double convolution 3D with batch normalization
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

#Double convolution with batch normalization
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

#First layer
class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()

        self.conv = DoubleConv3D(in_channels, out_channels, kernel)

    def forward(self, x):
        x = self.conv(x)
        return x

#Downsampling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        
        self.MaxPoolingConv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            DoubleConv3D(in_channels, out_channels, kernel)
        )
    
    def forward(self, x):
        x = self.MaxPoolingConv(x)
        return x
    
#Seq to image
class SeqToImage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.FC = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        
        x = x.permute(0,1,3,4,2)
        x = self.FC(x)
        x = x.permute(0,1,4,2,3)
        
        return x.squeeze(dim=2)

# Upsampling
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()

        if bilinear: # Standart
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:        # Upsample trainable
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, 2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel)
        #self.conv_skip = DoubleConv(in_channels, out_channels, kernel) #ajustar

    def forward(self, x1, skip):
        x1 = self.up(x1)
        
        # input is CHW
        '''diffY = skip.size()[2] - x1.size()[2]
        diffX = skip.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        '''
        #print(skip.shape)
        #print(x1.shape)

        #skip = skip.reshape(x1.shape[:2]+(1,)+x1.shape[:-2])
        #skip = self.conv_skip(skip) #ajustar
        
        #frames = skip.shape[2]
        #print(frames)
        #skip = torch.split(skip, 1, dim=2)
        #skip = skip[len(skip) // 2] #SEMPRE PEGAR O FRAME DO MEIO
        #skip = skip.squeeze(dim=2)

        #print('*'*10)
        #print(skip.shape)
        #print(x1.shape)

        x = torch.cat([skip, x1], dim=1)
        x = self.conv(x)
        return x

#Last layers
class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1):
        x = self.conv(x1)
        return x