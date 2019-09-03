# Face-Gender Prediction
This repository contains trainer for general classification case. However, this repo specifically use the trainer to train face gender prediction

At the moment, you can easily:  
 * Image cropping is done near-on-the-fly using TorchETL
 * Train and evaluate multiple datasets
 * Log training and evaluation beautifully
 * Do early stopping
 * Save model based on metrics improvement
 * Reproduce your research easily since training is done by feeding configuration file into the trainer


### Table of contents
- [Face-Gender Prediction](#face-gender-prediction)
    - [Table of contents](#table-of-contents)
    - [Installation](#installation)
    - [Usage](#usage)
      - [Loading pretrained models](#loading-pretrained-models)
      - [Example: train.py](#example-trainpy)
      - [Deeper look](#deeper-look)
    - [Deeper look](#deeper-look-1)

      - 

### Installation

Build docker container

### Usage

#### Loading pretrained models


#### Example: train.py
 
import argparse  
from yaml import safe_load  
from collections import defaultdict  
from typing import Dict, Callable, List, Sequence, Optional  
import re  
import os  
import pdb  

import torch  
import torchetl  

from nf_trainer.utils import elapsed_timer, create_instance, create_instance_dataloader, create_instance_metrics  
import nf_trainer.datasets as nf_datasets  
import nf_trainer.dataloaders as nf_dataloaders  
import nf_trainer.models as nf_models  
import nf_trainer.metrics as nf_metrics  
import nf_trainer.transforms as nf_transforms  
from nf_trainer.trainer import Trainer  

parser = argparse.ArgumentParser(description="Training pipeline for predicting gender based on face")  
parser.add_argument('-c', '--configuration_path' ,type=str, help='path to experiment configuration file', default='training_configs/exp41.yaml')  
parser.add_argument('-g', '--gpu_id', type=int, default=0)  

args = parser.parse_args()  

with open(args.configuration_path, 'r') as f:  
    try:  
        config = safe_load(f)  
    except FileNotFoundError:  
        print(f'File not found at {args.configuration_path}')  
        exit()  


device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")  
print(f'device {device}')  
 
model = create_instance(config['model'], nf_models)  

optimizer = create_instance(config['optimizer'], torch.optim, model.parameters())  

dataloader = create_instance_dataloader(config['partition'], nf_transforms, torchetl.etl, nf_dataloaders)  

loss = create_instance(config['loss'], torch.nn)  
metrics = create_instance_metrics(config['metrics'], nf_metrics)  

#do not connect to wandb during test run  
#os.environ['WANDB_MODE'] = 'dryrun'  
create_instance(config['wandb'], wandb)  
wandb.watch(model)  

trainer = Trainer(config['trainer'], model, dataloader, loss, metrics, optimizer, device)  
trainer.train()  


#### Deeper look

### Deeper look

Here is the structure of the library
- nodeflux
    - architecture
    - dataloaders
    - datasets
    - lib
    - metrics
    - models
    - prepare
    - architecture
    - transforms

architecture folder stores model architecture / backbone such as efficientnet and mobilenet. If you want to use other architectures, make sure to put them
in this folder  
dataloaders folder store dataloaders inherited from torch.utils.data.DataLoader
datasets folder store datasets inherited from torch.utils.data.Dataset
lib folder store trained models that inherits from torch.nn.Models
metrics folder store 

