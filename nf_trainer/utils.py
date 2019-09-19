from torchetl.base.dataset import BaseDataset
from pathlib import Path, PosixPath
import numpy as np
import cv2
import pdb
from tqdm import tqdm
from torchetl.etl import TransformAndLoad
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time
from contextlib import contextmanager
from timeit import default_timer
import time
import importlib
from typing import Iterable
import pandas as pd
import csv
import nonechucks as nc



def compute_normalization_values(files: Iterable):
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

def create_instance(config_params: dict, module: object, **kwargs):
    """
    
    """
    print(f'Initializing module {module.__name__}')
    params = config_params
    params['args'].update(kwargs)
    # config_params['args'].update(kwargs)
    instance = getattr(module, params['module'])(**params['args'])
    print(f'create_instanced module {module.__name__}')
    return instance

def create_instance_dataloader(config_params, transforms_module, dataset_module, dataloader_module):
    """
    apply function 


    finds a function handle with the name given as 'module' in config_params, and returns the 
    instance create_instanced with corresponding keyword args given as 'args'.

    parameter
    config_params -> dictionary that contains key value pair of parameters
    module_name -> ex, see in test2.yaml, example dataloaders, optimizer, lr_scheduler
    path_to_module -> path to module containing the class

    """
    print(f'Initializing dataset and dataloader')
    dataloaders = {}
    for partition in config_params.keys():
        partition_key_value = config_params[partition]
        transforms = getattr(transforms_module, partition_key_value['transforms_module'])(**partition_key_value['transforms_args'])
        dataset_args = partition_key_value['dataset_args']
        dataset_args['transform'] = transforms
        dataset = getattr(dataset_module, partition_key_value['dataset_module'])(**partition_key_value['dataset_args'])
        # dataset = nc.SafeDataset(dataset)
        dataloader_args = partition_key_value['dataloader_args']
        dataloader_args['dataset'] = dataset
        dataloader = getattr(dataloader_module, partition_key_value['dataloader_module'])(**dataloader_args)
        dataloaders[partition] = dataloader
    
    print(f'create_instanced dataset and dataloader')
    return dataloaders


def create_instance_metrics(config_params: dict, module, *args, **kwargs):
    """
    run function or module 
    finds a function handle with the name given as 'module' in config_params, and returns the 
    instance create_instanced with corresponding keyword args given as 'args'.

    parameter
    config_params -> dictionary that contains config_params
    primary_key -> ex, see in test2.yaml, example dataloaders, optimizer, lr_scheduler
    module -> path to module containing the class

    """
    print(f'Initializing {config_params["module"]}')
    instances = [getattr(module, i) for i in config_params['module']]
    #instance = getattr(module, config_params[primary_key]['module'])(*args, **config_params[primary_key]['args'])
    print(f'create_instanced {config_params["module"]}')
    return instances



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

    
def dump_bbox_and_landmark_to_csv(face_detection, array, columns, parent_directory, save_to):
    bbox_and_landmark_matrix = np.zeros((array.shape[0], len(columns)-array.shape[1]))


    for index in tqdm(range(len(array))):
        image_path = parent_directory / array[index, 0]
        image_original = cv2.imread(str(image_path))
        image_blob = np.expand_dims(image_original, axis=0)
        try:
            face_blob, landmark_blob = face_detection.detect(image_blob)
            index_of_biggest_face = get_center_face_bbox_index(image_original, face_blob[0])
            bbox_and_confidence, landmark = face_blob[0][index_of_biggest_face], landmark_blob[0][index_of_biggest_face]
            bbox = bbox_and_confidence[:4]
            landmark = landmark.reshape(-1,)
            bbox_and_landmark_value = np.concatenate((bbox,landmark))

        except ValueError:
            bbox_and_landmark_value = np.zeros((1, 14))
            print(f"Frontal face not detected on path {image_path}")
            pass

        except cv2.error:
            bbox_and_landmark_value = np.zeros((1, 14))
            pass
        
        except AttributeError:
            bbox_and_landmark_value = np.zeros((1, 14))
            print(f"Face not detected on path {image_path}")

        
        bbox_and_landmark_matrix[index] = bbox_and_landmark_value

    metadata_matrix = np.c_[array, bbox_and_landmark_matrix]
    df = pd.DataFrame(data=metadata_matrix, columns=columns)
    df.to_csv(save_to, index=False)

    print(f'Sucessfully dump into {save_to}')


