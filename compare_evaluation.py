import os
import csv
import cv2
import pandas as pd
import numpy as np
import random

files = [
    ('mix_loss_300k_4', 'results/mix_loss_v3_adam/val_images/eval_300000_under_4.csv'),
    ('mix_loss_170k_4', 'results/mix_loss_v3_adam/val_images/eval_170000_under_4.csv'),
    ('mix_loss_300k_6', 'results/mix_loss_v3_adam/val_images/eval_300000_under_6.csv'),
    ('mix_loss_170k_6', 'results/mix_loss_v3_adam/val_images/eval_170000_under_6.csv'),
    ('mix_loss_300k_8', 'results/mix_loss_v3_adam/val_images/eval_300000_under_8.csv'),
    ('mix_loss_170k_8', 'results/mix_loss_v3_adam/val_images/eval_170000_under_8.csv')
]

print('Name;ssim_avg;ssim_std;psnr_avg;psnr_std')
for name, file in files:
    df = pd.read_csv(file, sep=';')

    ssim_avg = np.average(df['ssim'])
    ssim_std = np.std(df['ssim'])

    psnr_avg = np.average(df['psnr'])
    psnr_std = np.std(df['psnr'])

    print('{};{};{};{};{}'.format(name, ssim_avg, ssim_std, psnr_avg, psnr_std))