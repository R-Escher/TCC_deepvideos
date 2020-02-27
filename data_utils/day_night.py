import os
import numpy as np
from skimage import io, transform
import cv2

def change_video_name(path, avg):

	file_name = path.split('/')[-1]
	if ('d_' not in file_name) and ('n_' not in file_name):

		if avg > 63.0:
			new_name =  'd_' + file_name
		else:
			new_name =  'n_' + file_name

		os.rename(path, path.replace(file_name, new_name))
	

def read_video(video_path):
	cap = cv2.VideoCapture(video_path)

	while True:
		ret, frame = cap.read()
		
		if ret:
			avg = np.average(frame)
			
			change_video_name(video_path, avg)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			break
		else:
			break

	cap.release()
	cv2.destroyAllWindows()

def read_image(path):
	img = io.imread(path)
	medians.append(np.median(img))

if __name__ == '__main__':	
	path = '/media/albano/external'
	#path = './'
	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.mov'):
				read_video(os.path.join(root,f))



