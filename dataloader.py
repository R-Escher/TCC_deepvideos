import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import cv2
import skimage
from scipy import ndimage, misc
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

def BddDataloader(dataset, batch_size, num_workers, shuffle=True):
    
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_custom_collate,
            shuffle=shuffle)

def _custom_collate(batch):
    data = torch.stack([item['x'] for item in batch], dim=0)
    target = torch.stack([item['y'] for item in batch], dim=0)

    return {'x': data, 'y': target}


class BddDataset(Dataset):

    """
    Attributes
    ----------
    csv_path : str (required)
        csv with video url's
    exposure : ('under' or 'over') (required)
        exposition type
    batch_size : int (required)
        batch_size
    window_size : int
        size of the temporal window
    frames_per_video: int
        number of frames per video
    causality : bool
        target frame in the middle of the temporal window if False, else at the end
    offset: int
        offset in the temporal window
    sparsity : bool
        if true, progressive increase offset 
    """

    def __init__(self, csv_path, data_path, exposure, batch_size, window_size=3, frames_per_video=50, causality=False, offset=0, sparsity=False, validation=False):

        if validation:
            self.gamma = exposure
        elif exposure == 'under':
            self.gamma = [4, 6, 8]
        elif exposure == 'over':
            self.gamma = [0.25, 0.16, 0.125]
        else:
            sys.exit("Exposition type must be 'under' ou 'over'!")

        self.data_path = data_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_video = frames_per_video
        self.causality = causality
        self.video_path_loader = pd.read_csv(csv_path)
        self.n_videos = len(self.video_path_loader.index)
        self.validation = validation

    def __len__(self):
        return self.n_videos

    def __getitem__(self, idx):
        
        # calculate a random temporal window
        window_config = self._get_random_window_index()

        # get a random video url
        video_name = self.video_path_loader.iloc[idx, :].values[0]

        # get sample
        sample = self._get_sample(self.data_path+video_name, window_config)

        return sample

    def _get_random_window_index(self):
        
        target = 0  # target frame
        window = [] # auxiliary frames
        
        if self.causality == False: # target in the middle of the window
            offset = int(self.window_size/2)
            target = random.randrange(0+offset, self.max_video-offset) if self.validation == False else (0+offset)            
            for i in range(target-offset, target+offset+1):
                window.append(i)

        if self.causality == True: # target in the end of the window
            offset = (self.window_size - 1)
            target = random.randrange(0+offset, self.max_video) if self.validation == False else (0+offset)
            for i in range(target-offset, target+1):
                window.append(i)

        return {'target': target, 'aux': window}

    def _get_sample(self, video_path, window):
    
        # set transformations
        transform = transforms.Compose(self._transforms_list())
        gamma_value = random.choice(self.gamma)
                
        # load images
        window_paths = ['{}/{:02d}.png'.format(video_path, x) for x in window['aux']]
        auxiliaries = skimage.io.imread_collection(window_paths)
        #auxiliaries = [aux for aux in auxiliaries]
        gt = auxiliaries[window['aux'].index(window['target'])]
        
        # transform ground-truth
        gt = transforms.functional.to_pil_image(gt) # to image
        gt = transform(gt) # to tensor transformed

        # transform auxiliaries
        auxiliaries = [self._change_gamma(aux, gamma_value) for aux in auxiliaries] # change gamma value (exposition)
        auxiliaries = [transform(aux) for aux in auxiliaries] # to tensor transformed
        auxiliaries = torch.stack(auxiliaries, dim=1)# to 3d tensor
        
        # set sample
        sample = {
            'x': auxiliaries,
            'y': gt
        }

        return sample

    def _transforms_list(self):
        return [
            transforms.Resize((400, 720)),
            transforms.CenterCrop((400, 400)),
            #transforms.Lambda(lambda x: ndimage.rotate(x, 90, reshape=True)),
            transforms.ToTensor(),
        ]

    def _change_gamma(self, f, gamma):
        f = transforms.functional.to_pil_image(f)
        f = transforms.functional.adjust_gamma(f, gamma)

        return f