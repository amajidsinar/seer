# standardlib
import csv
import re
from pathlib import Path, PosixPath
from typing import List, Tuple, Callable, Optional, Dict
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

# user
from nodeflux.face_alignment.arcface_alignment import ArcFaceAlignment

__all__ = ["NodefluxDataset"]


class NodefluxDataset(Dataset):
    def __init__(self, 
                parent_directory: str, 
                csv_file: str,  
                split_key: str,
                label_type_key: str,
                dataset_name: str = None,
                label_encoding: Dict[str, int] = None,
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
        self.face_alignment = ArcFaceAlignment()
        self.label_type_key = label_type_key
        try:
            metadata_df = pd.read_csv(str(csv_file))
            # filter by split key
            metadata_df = metadata_df[metadata_df['split'] == split_key]
            # filter by label type
            metadata_df = metadata_df[metadata_df['label_type'] == label_type_key]
            # label_value
            #filter by dataset name
            if dataset_name:
                metadata_df["dataset_name"] = metadata_df['path'].apply(lambda x: Path(x).parts[0])
                metadata_df = metadata_df[metadata_df['dataset_name'] == dataset_name]

            if label_encoding:
                for current_label, desired_label in label_encoding.items():
                    metadata_df['label_name'].loc[metadata_df['label_name'] == current_label] = desired_label

            self.metadata_df = metadata_df.reset_index(drop=True)
            self.label_value = self.metadata_df['label_value'].apply(lambda x: x.split(' '))
        
            # TODO ADD FILTER FAILED DETECTION
            # landmark_df = metadata_df.iloc[:, landmark_column_index[0]:landmark_column_index[1]]
            # landmark_df = landmark_df.replace(0, np.nan)
            # landmark_df = landmark_df.dropna(how='all')
            # filtered_index = landmark_df.index
            # metadata_df = metadata_df.loc[filtered_index]
            # self.metadata_df = metadata_df.reset_index(drop=True)
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
        
        #image_path = self.parent_directory / self.metadata_df.iloc[0, idx]
        image_path = self.parent_directory / self.metadata_df['path'][idx]
        
        target = self.metadata_df['label_name'][idx]
        image_array = cv2.imread(str(image_path))
        label_value = np.array([float(i) for i in self.label_value[idx]])
        if self.label_type_key == "keypoint_detection":
            label_value = label_value.reshape((5,2))
            label_value = np.expand_dims(label_value, axis=0)


        image_array, _ = self.face_alignment.extract_aligned_faces(image_array, label_value)
        image_array = image_array[0]

        if self.transform:
            image_array = self.transform(image_array)

        return image_array, target
