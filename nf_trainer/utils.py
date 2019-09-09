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
        #dataset = nc.SafeDataset(dataset)
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

def create_object(config_params: dict, module):
    object = getattr(module, config_params['module'])
    print(f'create_instanced module {module.__name__}')
    return object

def store_hyperparams(config_params, hyperparam_name):
    hyperparams = {f'{hyperparam_name}': config_params['module']}
    for hyperparam_name, hyperparam_value in config_params['args'].items():
        hyperparams[hyperparam_name] = hyperparam_value
    return hyperparams


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


def dump_bbox_and_landmark_to_csv(face_detection: object, parent_directory: PosixPath, csv_file:PosixPath, save_to: str):
    print(f'Creating file in {save_to}')
    df = pd.read_csv(str(csv_file))
    path = df.iloc[:,0]
    label = df.iloc[:,1]


    header = ["path", "label"]
    for bbox_coordinate in ["min", "max"]:
        header += [f'bbox_x_{bbox_coordinate}', f'bbox_y_{bbox_coordinate}']
    for landmark_coordinate in range(5):
        header += [f'landmark_x_row_{landmark_coordinate}', f'landmark_y_row_{landmark_coordinate}']


    with open(save_to, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        for index in tqdm(range(len(df))):
            image_path = parent_directory / df.iloc[index, 0]
            image_original = cv2.imread(str(image_path))
            image_blob = np.expand_dims(image_original, axis=0)
            try:
                face_blob, landmark_blob = face_detection.detect(image_blob)
                index_of_biggest_face = get_center_face_bbox_index(image_original, face_blob[0])
                bbox_and_confidence, landmark = face_blob[0][index_of_biggest_face], landmark_blob[0][index_of_biggest_face]
                bbox = bbox_and_confidence[:4]
                row = [path[index]]
                row += [label[index]]
                for i in bbox:
                    row += [i]
                
                for row_index in range(landmark.shape[0]):
                    for column_index in range(landmark.shape[1]):
                        row += [landmark[row_index][column_index]]

                csv_writer.writerow(row)

            except ValueError:
                print(f"Frontal face not detected on path {path[index]}")
                pass

            except cv2.error:
                pass
            
            except AttributeError:
                print(f"Face not detected on path {path[index]}")


    print(f'Sucessfully dump into {save_to}')
    

