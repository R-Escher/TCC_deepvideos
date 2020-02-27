import os
import csv
import cv2
import pandas as pd
import random
import numpy as np

def insert_video_samples_in_csv(out_file_name, video_path, video_length, window_length=3, max_samples=40, fixe_window=False):

	samples = []

	for i in range(max_samples): #target frame per video

		target = i
		frames = ''

		for l in range(1, int(window_length/2)+1): #frame per window size
			if(i - l >= 0):
				frames = frames + str(i - l) + '-'
			if(i + l <= video_length):
				frames = frames + str(i + l) + '-'
		if fixe_window == False:
			samples.append([video_path, target, frames[:-1]])
		elif len(frames.split('-'))-1 == (window_length-1):
			#print(len(frames.split('-')))
			samples.append([video_path, target, frames[:-1]])

	with open(out_file_name, mode='a') as outfile:
		writer = csv.writer(outfile, delimiter=',')

		for sample in samples:
			writer.writerow(sample)

def write_video_path(out_file, video_path):
	with open(out_file, mode='a') as outfile:
		writer = csv.writer(outfile, delimiter=',')

		writer.writerow([video_path])

def read_video(path, n_frames=1):
	cap = cv2.VideoCapture(path)
	count = 0
	video_avg = []

	while True:
		ret, frame = cap.read()

		if ret:
			frame_avg = np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			video_avg.append(frame_avg)

			count += 1
			if count == n_frames: break

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	video_length = 100 #int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()

	return video_length, np.average(video_avg)

if __name__ == '__main__':

	out_path = './csv_loaders/bdd_day[90-110]_train_5k_40.csv'
	in_path = '/media/albano/external'
	max_videos = 6 * 1000 + 100

	with open(out_path, mode='w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(['video_path'])

	samples = []
	count = 0
	for root, dirs, files in os.walk(in_path):
		for f in files:
			if f.endswith('.mov') and f[0] == 'd':
				video_length, video_avg = read_video(os.path.join(root,f))
				if 90 <= video_avg <= 110: # optical_flow > x
					#insert_video_samples_in_csv(out_path, os.path.join(root,f), video_length, fixe_window=True)
					write_video_path(out_path, os.path.join(root,f))
					if count == max_videos: break
					else:  count += 1


	for df in pd.read_csv(out_path, sep=',', chunksize=1):
		#print(df['video_path'].item())
		#print(df['target_frame'].item())
		#print(df['frames_list'].item())
		#count = count + 1
		break
	#print(count)
