import os
import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
import csv

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
	out_path = './csv_loaders/bdd_day[90-110]_all.csv'
	in_path = '/media/albano/external'

	with open(out_path, mode='w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(['video_path'])

	samples = []
	count = 0
	for root, dirs, files in os.walk(in_path):
		for f in files:
			if f.endswith('.mov') and f[0] == 'd':
				video_length, video_avg = read_video(os.path.join(root,f))
				if 90 <= video_avg <= 110:
					write_video_path(out_path, os.path.join(root,f))