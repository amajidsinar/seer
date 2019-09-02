import torch
from nf_trainer.trainer import set_parameter_requires_grad
from efficientnet_pytorch.model import EfficientNet
from torch import nn


__all__ = ['EfficientNet_B0', 'EfficientNet_B1']

def EfficientNet_B0(use_pretrained, feature_extract, n_class, gpu_id, *args, **kwargs):
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    if use_pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b0')
            
    else:
        model = EfficientNet.from_name('efficientnet-b0')

    model._fc = nn.Linear(1280, n_class, bias=True)
    model.to(device)
    set_parameter_requires_grad(model, feature_extract)

    return model

def EfficientNet_B1(use_pretrained, feature_extract, n_class, gpu_id, *args, **kwargs):
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    # if gpu_id >= 0:
    #     device = torch.device(f'cuda:{gpu_id}')
        
    # else:
    #     device = torch.device('cpu')
    
    if use_pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b1')
    else:
        model = EfficientNet.from_name('efficientnet-b0')
            
    model._fc = nn.Linear(1280, n_class, bias=True)
    model.to(device)
    set_parameter_requires_grad(model, feature_extract)

    return model


