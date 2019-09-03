# nf_trainer
This repository contains trainer for image classification tasks. However, it is highly possible to extend this repository for other tasks, for instance object detection


At the moment, you can easily:  
 * Train from configuration file, making it much easier to reproduce
 * Train and evaluate on multiple datasets
 * Log training and evaluation metrics, along with the hyperparameters
 * Early stop
 * Save weights based on selected metrics
  

### Table of contents
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Deeper look at train.py](#deeper-look-at-trainpy)
    - [Deeper look at example configuration file](#deeper-look-at-example-configuration-file)
  - [Structure](#structure)


### Installation

pip3 install -r requirements.txt

### Getting Started

It's hard to fully exploit this trainer without knowing it first

#### Deeper look at train.py

Navigate to train.py and notice that 


#### Deeper look at example configuration file

### Structure

Here is the structure of the library
- nf_trainer
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

