# standardlib
import csv
import re
from pathlib import Path, PosixPath
from typing import List, Tuple, Callable, Optional
from collections import namedtuple
import pdb

# external package
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from skimage import transform as trans
from nodeflux.face_detection.retinaface import RetinaFace

# user
from nodeflux.face_alignment.arcface_alignment import ArcFaceAlignment

__all__ = ["FaceDataset"]


class FaceDataset(Dataset):
    def __init__(self, 
                parent_directory: str, 
                csv_file: str, 
                bounding_box_column_index: List[int],
                landmark_column_index: List[int], 
                transform: Callable = None,
                apply_face_cropping : bool = False,
                apply_face_alignment: bool = False,
                ) -> None:
        """Class for reading csv files of train, validation, and test
        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory
        csv_file
            The path to csv file containing the information of the image. For the bare minimum it should contain path and label.
            If apply_face_cropping is set to True, then it must contain bounding box for index 2 until 5
            If apply_face_alignment is set to True, then it must contain bounding box for index 6 until 15
        transform
            Callable which apply transformations
        apply_face_cropping
            Read description in csv_file
            In addition, if apply_face_cropping is set to True, then apply_face_alignment must be set to False
        resize_to
            Resize input image to this value. Always set this value. By default is valued at (640,480)
        apply_face_alignment
            Read description in csv_file
            In addition, if apply_face_alignment is set to True, then apply_face_cropping must be set to False
        Returns
        -------
        None	
        """
        self.parent_directory = Path(parent_directory)
        self.transform = transform
        self.apply_face_cropping = apply_face_cropping
        self.apply_face_alignment = apply_face_alignment
        self.bounding_box_column_index = bounding_box_column_index
        self.landmark_column_index = landmark_column_index
        self.face_alignment = ArcFaceAlignment()

        try:
            self.csv_file = pd.read_csv(str(csv_file))
        except FileNotFoundError:
            print(f'{Path.cwd() / csv_file} does not exist')

    def __len__(self) -> int:
        """Return the length of the dataset
        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory
        csv_filecsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)
            The path to csv file containcsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)ing X and y
        Transformcsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)
            Callable which apply transfocsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)rmations
        Returns
        -------
        Length of the dataset	
        """
        return len(self.csv_file)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the X and y of a specific instance based on the index
        Parameters
        ----------
        idx
            The index of the instance 
        Returns
        -------
        Tuple of X and y of a specific instance	
        """
        image_path = self.parent_directory / self.csv_file.iloc[idx, 0]
        target = self.csv_file.iloc[idx, 1]
        image_array = cv2.imread(str(image_path))
        
        if self.apply_face_cropping and self.bounding_box_column_index:
            assert not self.apply_face_alignment
    
            bounding_box_index_start, bounding_box_index_end = self.bounding_box_column_index
            x_min, y_min, x_max, y_max = self.csv_file.iloc[idx, bounding_box_index_start:bounding_box_index_end+1].astype(int)
            image_array = image_array[y_min:y_max, x_min:x_max]

        if self.apply_face_alignment and self.landmark_column_index:
            assert not self.apply_face_cropping
            
            image_landmark = self.csv_file.iloc[idx, self.landmark_column_index[0]:self.landmark_column_index[1]+1].values
            image_landmark = image_landmark.reshape((5,2))
            blobs_landmark = np.expand_dims(image_landmark, axis=0)
  
            image_array, _ = self.face_alignment.extract_aligned_faces(image_array, blobs_landmark)
            image_array = image_array[0]

        if self.transform:
            image_array = self.transform(image_array)

        return image_array, target