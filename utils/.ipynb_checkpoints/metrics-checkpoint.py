import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

def calc_metrics(out, y):

    # swap axes, move to cpu, break gradient and cast to numpy array for metric calculation
    y = np.moveaxis(y.cpu().detach().numpy(), 1, -1)
    out = np.moveaxis(out.cpu().detach().numpy(), 1, -1)

    values = []
    # calc metrics over the batch
    for gt, predict in zip(y, out):
        calc_ssim = ssim(gt, predict, multichannel=True)
        calc_psnr = psnr(gt, predict)

        values.append([calc_ssim, calc_psnr])
    
    return values