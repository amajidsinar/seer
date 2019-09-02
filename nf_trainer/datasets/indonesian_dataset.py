from pathlib import Path
from torchvision import transforms
from torchetl.base.dataset import BaseDataset
from torchetl.etl import Extract
import pandas as pd
import pdb
from collections import namedtuple
import numpy as np
import re

class IndonesianDataset(BaseDataset):
    def __init__(self, parent_directory, extension, csv_file):
        super().__init__(parent_directory, extension)
        try:
            self.csv_file = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f'Csv file not found at {csv_file}')        

    def create_dataset_array_from_csv(self, verbose):
        """Create full dataset array from reading files
        Parameters
        ----------
        None
        Returns
        -------
        Tuple of X and y	
        """
        Dataset = namedtuple('Dataset', ['filename', 'target'])

        target = []
        filename = []
        # absolute path leading to file
        for full_path in self.read_files():
            parent_length = len(self.parent_directory.parts)
            relative_path = full_path.parts[parent_length:]
            person_name = relative_path[0]
            match_row = (self.csv_file[self.csv_file["NAMA"] == str(person_name)])
            target_label = match_row["JENIS_KELAMIN"].values
            for encoded_label, label in enumerate(self.labels):
                if bool(label == target_label):
                    target.append(encoded_label)
            filename.append(str(Path(*relative_path)))
            
        if verbose:
            print("Finished creating whole dataset array")

        dataset = Dataset(np.array(filename), np.array(target))
        return dataset