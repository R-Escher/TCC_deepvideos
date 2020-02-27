import os
import sys
sys.path.append('/ucan_utils')

import cv2
import keras
from keras.models import load_model
import numpy as np 
from ucan_utils import loss_definition

def getPath(path):
    #path = './a/b'

    path = path.split('/')
    folder =  path.pop(-1)
    path = '/'.join(path)
    if path != '':
        path = path + '/'

    return path, folder

def loadModels():

    # Load under model
    model_path = 'ucan_utils/model_under.hd5f'
    weights_path = 'ucan_utils/weigths_under.hd5f'
    under_model = load_model(model_path, custom_objects={'loss_mix_v3': loss_definition.loss_mix_v3})
    under_model.load_weights(weights_path)

    # Load over model
    model_path = 'ucan_utils/model_over.hd5f'
    weights_path= 'ucan_utils/weigths_over.hd5f'
    over_model = load_model(model_path, custom_objects={'loss_mix_v3': loss_definition.loss_mix_v3})
    over_model.load_weights(weights_path)

    return under_model, over_model

def ucan(rgb, model):

    rgb = rgb/255.

    # Add one dimmension -> (1, width, height, channels)
    shape = rgb.shape
    rgb = np.reshape(rgb, (1, shape[0],shape[1], shape[2]))

    # Model predict
    out = np.clip(model.predict(rgb), .0, 1.)

    # Remove one dimmension -> (width, height, channels)
    shape = out.shape
    out = np.reshape(out, (shape[1], shape[2], shape[3]))

    return (out*255).astype(np.uint8)  

def process(path, model, exposure, out_ext='jpg'):
    
    # get name files in path
    files = [file for file in sorted(os.listdir(path)) if file.split('.')[-1] == out_ext]

    # get np frames by opencv
    frames = [cv2.imread(path+'/'+file) for file in files]

    # get ucan result for each frame
    results = [ucan(frame, model) for frame in frames]

    # set save path
    path, folder = getPath(path)
    save_path = path+'/'+folder+'_'+exposure
    os.mkdir(save_path)

    # save new frames
    for result in results:
        file_name = str(results.index(result))+'.'+out_ext
        cv2.imwrite(save_path+'/'+file_name,result)


if __name__ == '__main__':

    under_model, over_model = loadModels()

    path = "./bdd_test/00"

    process(path, under_model, exposure='under')
    process(path, over_model, exposure='over')
