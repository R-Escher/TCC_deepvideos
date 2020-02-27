# Libs import
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

# My imports
from architectures import UNet3D, UNet
from dataloader import BddDataset, BddDataloader
from train import test_model as val_model
from loss import LossFunction
from utils import log

# Hiperparameters and configurations
RUN_NAME = 'fc_mix_7_3_teste'
RESULTS_PATH = 'results/'
#RUN_PATH = RESULTS_PATH+RUN_NAME+'/val_images/'
RUN_PATH = 'results/'+RUN_NAME+'/'
SEED = 12
BATCH_SIZE = 2
DATA_PATH = '~/Documents/bdd_images/'
VAL_FILE_PATH = DATA_PATH + 'bdd_day_val.csv'
DEGRADATION = 'over'
EXPOSURE = [0.25]
WINDOW_SIZE = 7
MODEL_STATE_NAME = '188000'
MODEL_STATE_PATH = RUN_PATH+'fc_mix_7_3'+'_'+MODEL_STATE_NAME+'.pth'

# Set host or device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()

# Log in file
sys.stdout = open('{}eval_{}_{}_{}.csv'.format(RUN_PATH, MODEL_STATE_NAME, DEGRADATION, EXPOSURE[0]), 'w')

# Set seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Set dataloaders
val_dataset = BddDataset(VAL_FILE_PATH, DATA_PATH, EXPOSURE,
                           BATCH_SIZE, window_size=WINDOW_SIZE, validation=True)
val_loader = BddDataloader(val_dataset, BATCH_SIZE, num_workers=4, shuffle=False)

# Set model
model = UNet3D.UNet3D(WINDOW_SIZE).to(device)
#model = UNet.UNet(3, 3).to(device)
model.load_state_dict(torch.load(MODEL_STATE_PATH))

# Set optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())

# Set criterion
criterion = LossFunction().to(device)

print('ssim;psnr')
# Iterate over val loader
for _, sample in enumerate(val_loader):

    # Send data to device
    x = sample['x'].to(device=device, dtype=torch.float)
    y = sample['y'].to(device=device, dtype=torch.float)
    
    x = torch.squeeze(x, 2) if WINDOW_SIZE == 1 else x
        
    # val model with sample
    out, loss = val_model(model, {'x': x, 'y': y}, criterion, optimizer)

    # swap axes, move to cpu, break gradient and cast to numpy array for metric calculation
    y = np.moveaxis(y.cpu().detach().numpy(), 1, -1)
    out = np.moveaxis(out.cpu().detach().numpy(), 1, -1)

    # calc metrics over the batch
    for gt, predict in zip(y, out):
        calc_ssim = ssim(gt, predict, multichannel=True)
        calc_psnr = psnr(gt, predict)

        print('{};{}'.format(calc_ssim, calc_psnr))