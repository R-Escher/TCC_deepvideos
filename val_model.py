# Libs import
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils

# My imports
from architectures import UNet3D
from dataloader import BddDataset, BddDataloader
from train import train_model, test_model
from loss import LossFunction
from utils import log

RUN_NAME = 'fc_mix_3_3'
RESULTS_PATH = 'results/'
#RUN_PATH = RESULTS_PATH+RUN_NAME+'/val_images/'
RUN_PATH = 'results/'+RUN_NAME+'/'
SEED = 12
BATCH_SIZE = 1
DATA_PATH = '/home/albano/Documents/bdd_images/'
VAL_FILE_PATH = DATA_PATH + 'bdd_day_val.csv'
DEGRADATION = 'over'
EXPOSURE = [0.16]
WINDOW_SIZE = 3
MODEL_STATE_NAME = '47000'
MODEL_STATE_PATH = RUN_PATH+'weights/'+'fc_mix_3_3'+'_'+MODEL_STATE_NAME+'.pth'
SAVE_IMAGES_PATH = 'qualitative_results/3_over/'

# Set host or device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set dataloaders
val_dataset = BddDataset(VAL_FILE_PATH, DATA_PATH, EXPOSURE,
                           BATCH_SIZE, window_size=WINDOW_SIZE, validation=True)
val_loader = BddDataloader(val_dataset, BATCH_SIZE, num_workers=4, shuffle=False)


# Set model and load weights
model = UNet3D.UNet3D(WINDOW_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_STATE_PATH))

# Set optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())

# Set criterion
criterion = LossFunction().to(device)

val_loss = []
#rint('Video Loss')

# Iterate over videos.
for video_step, sample in enumerate(val_loader):
    # Iterate over frames.

    # Send data to device
    x = sample['x'].to(device=device, dtype=torch.float)
    y = sample['y'].to(device=device, dtype=torch.float)

    # Test model with sample
    out, loss = test_model(model, {'x': x, 'y': y}, criterion, optimizer)
    #val_loss.append(loss)
    #print('{} {}'.format(video_step, loss))

    log.log_images(x, y, out, '{}{}_'.format(SAVE_IMAGES_PATH, video_step), BATCH_SIZE, WINDOW_SIZE)
