import torch
from torchvision import utils
import time
import datetime
import numpy as np


def log_time(msg):
    print(msg)
    print('\t', end='')
    print('Datetime: {}'.format(datetime.datetime.now()), end='\n')


def log_images(x, y, out, path, batch_size, window_size):

    with torch.no_grad():
        
        if window_size == 1:
            frames = [x.cpu()]
        else:
            frames = torch.split(x.cpu(), 1, dim=2)
            frames = [frame.squeeze(dim=2) for frame in frames]
        frames.append(out.cpu())
        frames.append(y.cpu())

        # make a single list with all frames
        to_log = []
        for i in range(batch_size):
            for f in frames:
                to_log.append(f[i])

        # concat all in one dimension
        to_log = torch.stack(to_log)

        # make a grid, where each row is a sample (5 frames)
        grid = utils.make_grid(to_log, nrow=window_size+2)
        utils.save_image(grid, path + 'sample.png')


def log_model_eval(model):
    print('Model evaluation: ', model.eval())


def log_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Model trainable parameters: ', params)


def log_config(save_path, **kwargs):

    with open('{}config.txt', "w") as file:
        file.write('Datetime: {}'.format(datetime.datetime.now()), end='\n\n')
        for key, value in kwargs.items():
            file.write('{}: {}'.format(key, value))