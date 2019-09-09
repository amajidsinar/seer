#standardlib
import argparse
from yaml import safe_load
from collections import defaultdict
from typing import Dict, Callable, List, Sequence, Optional
import re
import pdb
from pathlib import Path

#external package
import comet_ml
import torch
import torchetl

# user
from nf_trainer.utils import elapsed_timer, create_instance, create_instance_dataloader, create_instance_metrics, store_hyperparams
import nf_trainer.datasets as nf_datasets
import nf_trainer.dataloaders as nf_dataloaders
import nf_trainer.models as nf_models
import nf_trainer.metrics as nf_metrics
import nf_trainer.transforms as nf_transforms
from nf_trainer.trainer import Trainer

parser = argparse.ArgumentParser(description="Training pipeline for predicting gender based on face")
parser.add_argument('-c', '--configuration_path' ,type=str, help='path to experiment configuration file', default='training_configs/exp41.yaml')
parser.add_argument('-r', '--resume', type=str, help='path to checkpoint file', required=False)
parser.add_argument('-g', '--gpu_id', type=int, default=0)

args = parser.parse_args()
with open(args.configuration_path, 'r') as f:
    try:
        config = safe_load(f)
    except FileNotFoundError:
        print(f'File not found at {args.configuration_path}')
        exit()

config['model']['args']['gpu_id'] = args.gpu_id
device = torch.device(f"cuda:{config['model']['args']['gpu_id']}" if config['model']['args']['gpu_id'] >= 0 else "cpu")
print(f'device {device}')

model = create_instance(config['model'], nf_models)

optimizer = create_instance(config['optimizer'], torch.optim, params=model.parameters())

try:
    lr_scheduler = create_instance(config['lr_scheduler'], torch.optim.lr_scheduler, optimizer=optimizer)
except KeyError:
    lr_scheduler = None
            
dataloaders = create_instance_dataloader(config['partition'], nf_transforms, torchetl.etl, nf_dataloaders)

loss = create_instance(config['loss'], torch.nn)

metrics = create_instance_metrics(config['metrics'], nf_metrics)

# do nor connect to wandb during test run

if args.resume:
    checkpoint = torch.load(args.resume)
    experiment_key = checkpoint['experiment_key']
    logger = create_instance({
        "module": config['comet_ml']['module_existing_experiment'],
        "args": config['comet_ml']['args']
    }, comet_ml, previous_experiment = experiment_key)
else:
    logger = create_instance({
        "module": config['comet_ml']['module_experiment'],
        "args": config['comet_ml']['args']
    }, comet_ml)

model_name = Path(args.configuration_path).stem
logger.set_name(model_name)
print(f'Training model {model_name}')

trainer = Trainer(
    model_name = model_name,
    trainer_configuration = config['trainer'], 
    model = model, 
    dataloaders = dataloaders, 
    loss = loss, 
    metrics = metrics, 
    optimizer = optimizer, 
    lr_scheduler = lr_scheduler,
    device = device, 
    logger = logger, 
    resume = args.resume)
 
trainer.train()
