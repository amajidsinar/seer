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
    - [A peek at train.py](#a-peek-at-trainpy)
    - [A peek at example configuration file](#A-peek-at-example-configuration-file)
    - [On a side note](#On-a-side-note)
  - [Structure](#structure)
  - [FAQ](#faq)


### Installation

```
pip3 install -r requirements.txt
```

### Getting Started

It's hard to fully exploit this trainer without understand its power and hence, limitations

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

The configuration file follows a specific pattern since we need to have key value pairs of module and args to instantiate objects. With a closer look, some dictionaries does not obey this rule: trainer, comet_ml, partition, and metrics. However, they still adhere similar pattern to what we have seen before.

In addition, if an object is instantiated from using create_instance function, then

#### On a side note

You don't have to use YAML, since the input to create instance is a dictionary. However, YAML is really intuitive so why not?
Configuration file name is very important, since the model name is derived from configuration file

### Structure

Here is the structure of the library
```
- nf_trainer
    - architecture
    - dataloaders
    - datasets
    - metrics
    - models
    - architecture
    - transforms
```
Make sure to put the correct stuff into the correct place. Currently, we support the following
```
- architecture
    - EfficientNet-B0
    - EfficientNet-B1
    - MobileNet_v3
- dataloaders
    - BaseDataLoader
- metrics
    - Accuracy
    - Precision
    - Recall
    - F1
- models
    - EfficientNet_B0
    - EfficientNet_B1
    - MobileNet_v3
    - ShuffleNet_v2_x1_0
- transforms
    - face

```

### FAQ
How can I develop the yaml?   
Follow the following pattern and remember that yaml can be parsed just like ordinary dictionary.

```
key:
  module: YOUR-OBJECT-NAME
  args:
    ARGS-NAME-1: ARGS-VALUE-1
    ARGS-NAME-2: ARGS-VALUE-2
    ARGS-NAME-3: ARGS-VALUE-3
```

How can I instantiate new function from the dictionary above?  

```
new_function = create_instance(
  config_params = YAML-FILE['key'],
  module = importlib(PATH-TO-MODULE),
) 
```






