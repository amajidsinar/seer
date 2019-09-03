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
    - [The grand idea](#the-big-idea)
    - [A peek at train.py](#deeper-look-at-trainpy)
    - [A peek at example configuration file](#deeper-look-at-example-configuration-file)
  - [Structure](#structure)


### Installation

```
pip3 install -r requirements.txt
```

### Getting Started

It's hard to fully exploit this trainer without understand its power and limitations

#### The grand idea

Ultimately, we want to preserve any hyperparameters into some file for the sake of reproducibility. There are many ways to do this, but to make things short lets just take two example.

The first example is to direcly put hyperparameter and its value in a configuration file without any kind of structure. While this looks cool at first, it is not. 

. Lets say we want to compare two optimizer, SGD and Adam.  

```
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  
```

```
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)  
```

Now things get really interesting. In order to be able to accomodate both optimizers, we must dump both hyperparameters into the file. If we want to include Rmsprop, things get really, really messy and hard to maintain.

```
torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

In conclusion, it is not a good idea to directly dump hyperparameters into a file. 

To overcome this, we should design trainer that takes in instantiated objects, where the objects are created from configuration file which obeys a 
specific pattern. By doing this, we would have configuration file and trainer that are more easier to maintain

#### A peek at train.py

Notice that instance creation is done using create_instance function. 

```
create_instance(config_params: dict, module: object, **kwargs)
```

The config_params must have module and args key, where the value are the function name and function arguments

```
optimizer:
  module: SGD
  args:
    lr: 0.01
    momentum: 0
```

The reason why we use kwargs argument is that sometimes the input parameters is another object that we have instantiated beforehand. The usage of kwargs
forces us to use keyworded argument instead of positional arguments


For instance, this is the documentation to instansiate SGD
```
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)  
```


Since we have params object somewhere up in the train.py, it would be nice if we account for this update without changing the configuration file

```
optimizer = create_instance(config['optimizer'], torch.optim, params=model.parameters())
```




#### A peek at example configuration file

The only 

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

