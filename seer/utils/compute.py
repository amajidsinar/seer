from typing import Iterable
from tqdm import tqdm
import cv2
from contextlib import contextmanager
from timeit import default_timer
import numpy as np

def normalization_values(files: Iterable):
    mean, std = 0., 0.

    for index, file in tqdm(enumerate(files, start=1)):
        # opencv reads to BGR instead to RGB
        file = cv2.imread(str(file)) / 255
        file = file.reshape(1,3,-1)
        mean += np.mean(file, axis=2)
        std += np.std(file, axis=2)
    
    # convert BGR to RGB
    mean, std = mean[0][::-1], std[0][::-1]
    result = {}
    result["mean"] = mean / (index+1)
    result["std"] = std / (index+1)
    
    return result
        
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def get_center_face_bbox_index(img, face_boxes):

	'''
	Function to obtain the center face bounding box
​
	:param face_boxes: The face boxes coordinates, output frame face_detector
					in numpy format
	:type face_boxes: numpy array
	:return: The largest face bounding box coordinate
	:rtype: numpy array
	'''
	img_size = np.asarray(img.shape)[0:2]

	if face_boxes.shape[0] == 1:
		selected_index = 0
	else:
		bounding_box_size = (
			face_boxes[:, 2]-face_boxes[:, 0])*(face_boxes[:, 3]-face_boxes[:, 1])
		img_center = img_size / 2
		offsets = np.vstack(
			[(face_boxes[:, 0]+face_boxes[:, 2])/2-img_center[1], (face_boxes[:, 1]+face_boxes[:, 3])/2-img_center[0]])
		offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
		# some extra weight on the centering
		selected_index = np.argmax(
			bounding_box_size-offset_dist_squared*2.0)

	return selected_index