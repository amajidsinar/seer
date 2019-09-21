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

__all__ = ["FaceLandmarkDataset"]


class FaceLandmarkDataset(Dataset):
    def __init__(self, 
                parent_directory: str, 
                csv_file: str, 
                partition: str,
                landmark_column_index: List[int], 
                transform: Callable = None,
                ) -> None:
        """Class for reading csv files of train, validation, and test
        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        metadata_df
            The path to csv file containing the metadata of the image.
            Must contain path, label, and, its landmarks  
        landmark_column_index
            Column in which the landmarks are located    
        transform
            Callable to apply the transformations
        -------
        None	
        """
        self.parent_directory = Path(parent_directory)
        self.transform = transform
        self.landmark_column_index = landmark_column_index
        self.face_alignment = ArcFaceAlignment()
        try:
            metadata_df = pd.read_csv(str(csv_file))
            metadata_df = metadata_df[metadata_df['partition'] == partition]
            landmark_df = metadata_df.iloc[:, landmark_column_index[0]:landmark_column_index[1]]
            landmark_df = landmark_df.replace(0, np.nan)
            landmark_df = landmark_df.dropna(how='all')
            filtered_index = landmark_df.index
            metadata_df = metadata_df.loc[filtered_index]
            self.metadata_df = metadata_df.reset_index(drop=True)
        except FileNotFoundError:
            print(f'{Path.cwd() / csv_file} does not exist')

    def __len__(self) -> int:
        """Return the length of the dataset
        Parameters
        ----------
        Returns
        -------
        Length of the dataset	
        """
        return len(self.metadata_df)

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
        image_path = self.parent_directory / self.metadata_df.iloc[idx, 0]
        target = self.metadata_df.iloc[idx, 1]
        image_array = cv2.imread(str(image_path))
    
        image_landmark = self.metadata_df.iloc[idx, self.landmark_column_index[0]:self.landmark_column_index[1]+1].values
        image_landmark = image_landmark.reshape((5,2))
        blobs_landmark = np.expand_dims(image_landmark, axis=0)

        image_array, _ = self.face_alignment.extract_aligned_faces(image_array, blobs_landmark)
        image_array = image_array[0]

        if self.transform:
            image_array = self.transform(image_array)

        return image_array, target