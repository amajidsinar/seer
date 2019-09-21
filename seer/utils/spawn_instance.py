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

def create_instance_dataloader(config_params: dict, transforms_module, dataset_module, dataloader_module):
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
        # dataset = nc.SafeDataset(dataset)
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