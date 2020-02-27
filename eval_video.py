import os
import sys
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import skimage.exposure
import torch
import random
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from architectures import UNet3D, UNet
from global_motion import sparse_optical_flow

def pil_to_cv(frames):
    return [np.array(frame)[:, :, ::-1].copy() for frame in frames]

def transforms_list():
    return [
        transforms.ToPILImage(),
        transforms.Resize((400, 720)),
        transforms.CenterCrop((400, 400)),
        #transforms.Lambda(lambda x: ndimage.rotate(x, 90, reshape=True)),
        transforms.ToTensor(),
    ]
gamma_index = 0
def change_gamma(f, gamma):
    global gamma_index
    
    f = transforms.functional.to_pil_image(f)
    f = transforms.functional.adjust_gamma(f, [4,6,8][gamma_index] if gamma == 'under' else [0.25, 0.16, 0.125][gamma_index])
    f = transforms.functional.to_tensor(f)
    
    gamma_index += 1
    if gamma_index == 3:
        gamma_index = 0
    
    # f = skimage.exposure.adjust_gamma(f, gamma)

    return f

def read_video(video_path, data_path, window_size, n_frames, gamma, dilatation=1):
    
    transform = transforms.Compose(transforms_list())
    windows = []
    targets = []
    offset = int(window_size/2) + int(dilatation/2)

    for target in range(offset, n_frames-offset+1):
        window = []
        for i in range(target-offset, target+offset+1, dilatation):
            window.append(i)
        windows.append(window)
        targets.append(target)
    # print(len(windows))

    video = []
    for t, w in zip(targets, windows):
        video.append({
            'target': ['{}/{:02d}.png'.format(video_path, t)], 
            'window': ['{}/{:02d}.png'.format(video_path, x) for x in w]
            })

    # print(video)

    for v in video: 
        v['target'] = [transform(frame) for frame in io.imread_collection([data_path + x for x in v['target']])]
        v['window'] = [change_gamma(transform(frame), gamma) for frame in io.imread_collection([data_path + x for x in v['window']])]
    
    return video

def restore(frames, window_size, dilatation, gamma, model_path, batch_size=6):
    
    MODEL_STATE_PATH = model_path
    device = 'cuda'
    
    model = UNet3D.UNet3D(window_size).to(device)
    model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=torch.device(device)))
    #optimizer = torch.optim.Adam(model.parameters())

    outputs = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            #_in = torch.stack([torch.stack([transforms.functional.to_tensor(x) for x in f], dim=1) for f in frames[i:i+batch_size]])
            _in = torch.stack([torch.stack([x for x in f], dim=1) for f in frames[i:i+batch_size]])
            #print(_in.shape)
            out = model(_in.to(device)).cpu()
            
            #print(out.shape)
            
            #outputs += [F.to_pil_image(o) for o in out]
            outputs += [o for o in out]

    return outputs

def avg_difference(frames):
    previous = np.average(frames[0])
    
    out = []
    for i in range(1, len(frames)):
        actual = np.average(frames[i])
        out.append(previous-actual)
        
        previous = actual
        
    return out

def save_videos(videos, path, name):
    
    
    # make a grid, where each row is a sample (5 frames)
    grid = utils.make_grid(videos, nrow=int(videos.shape[0]/3))
    utils.save_image(grid, path + name)

def eval_videos(videos, data_path, window_size, n_frames, dilatation, gamma, model_path):
    
    n_videos = videos.shape[0]
    #results = {'original':np.zeros(n_frames), 'exposed':np.zeros(n_frames), 'restored':np.zeros(n_frames)}
    results = {'original':np.zeros(n_frames-2), 'exposed':np.zeros(n_frames-2), 'restored':np.zeros(n_frames-2)}

    
    index = [200, 100, 400, 500, 600, 700, 800, 900, 1000]
    for i, path in enumerate(videos.iterrows()):
        #print(i)
        if i in index:
            path = path[1].values[0]

            video = read_video(path, data_path, window_size, n_frames, gamma, dilatation)
            
            original = [v['target'][0] for v in video]
            exposed = [change_gamma(video, gamma) for video in original]
            #print(video[0]['window'][0].shape)
            #print(exposed[0].shape)
            
            #to_restore = []
            #for v in original:
            #    aux = []
            #    for x in v['window']:
            #        aux.append(change_gamma(x, gamma))
            #    to_restore.append(torch.stack(aux))
                    
            #to_restore = torch.stack(to_restore)
            #print(to_restore.shape)
            
            restored = restore([v['window'] for v in video], window_size, dilatation, gamma, model_path)
    
        # calc optical flow
        
        #results['original'] += np.array(sparse_optical_flow(pil_to_cv(original)))
        #results['exposed'] += np.array(sparse_optical_flow(pil_to_cv(exposed)))
        #results['restored'] += np.array(sparse_optical_flow(pil_to_cv(restored)))
        
        #calc image avg difference
        #print('aqui1')
        #results['original'] += avg_difference(original)
        #print('aqui2')
        #results['exposed'] += avg_difference(exposed)
        #print('aqui3')
        #results['restored'] += avg_difference(restored)
            original = torch.stack(original)
            exposed = torch.stack(exposed)
            restored = torch.stack(restored)
            
            #print(original.shape)
            #print(exposed.shape)
            #print(restored.shape)


            #out = torch.stack([original, exposed, restored])
            out = torch.cat((original, exposed, restored), 0)

            #print(out.shape)
            save_videos(out, 'qualitative_results/videos/', '{}_{}.png'.format(gamma, i))
        
#     for key in results:
#         results[key] /= (n_videos-1)
    
#     return results

if __name__=="__main__":

    data_path = '~/Documents/bdd_images/'
    csv_path = data_path + 'bdd_day_val.csv'
    models_path = 'models/tcc_results/'
    model = models_path + 'fc_mix_3_3_over/fc_mix_3_3_over_154000.pth'
    #model = models_path + 'fc_mix_3_3_under/fc_mix_3_3_under_108000.pth'
    exposition = 'over'
    max_videos = 2000
    window_size = 3
    n_frames = 45
    
    results = eval_videos(pd.read_csv(csv_path, nrows=max_videos), data_path, window_size, n_frames, 1, exposition, model)
    
#     for key in results:
#         print(key, end=';')
#         for value in results[key]:
#             print(value, end=';')
#         print()