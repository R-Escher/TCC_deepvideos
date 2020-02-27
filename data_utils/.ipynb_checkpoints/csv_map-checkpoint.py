import os
import cv2
import csv
import pandas as pd
import numpy as np
from skimage import io, transform

def read_video(video_path, n_frames=50, out_path='/media/albano/bdd100k_images/'):

    # cria um novo folder para o vídeo
    video_name = video_path.split('/')[-1].split('.')[-2]
    out_path = out_path + video_name + '/'
    os.mkdir(out_path)
    
    # abre o vídeo
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()

        if ret:
            # preprocessa o frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform.resize(frame, (720, 405))
            frame = transform.rotate(frame, 90, resize=True)
            
            # salva o frame
            io.imsave("{}{:02d}.png".format(out_path, count), frame)
                        
            if count == (n_frames - 1):
                break
            else:
                count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()

if __name__ == '__main__':
    csv_path = './csv_loaders/bdd_day[90-110]_all.csv'
    max_videos = 10000
    
    video_df = pd.read_csv(csv_path)

    for i in range(max_videos):
        video_path = video_df.sample(n=1, replace=False)
        read_video(video_path.values[0][0])