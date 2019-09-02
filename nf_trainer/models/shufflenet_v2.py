import torch
from nf_trainer.trainer import set_parameter_requires_grad
from torch import nn
from nf_trainer.utils import create_instance
import importlib
from torchvision import models

__all__ = ['ShuffleNet_v2_x1_0']

def ShuffleNet_v2_x1_0(use_pretrained, feature_extract, n_class, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    model = models.shufflenet_v2_x1_0(use_pretrained)
    model.fc = nn.Linear(in_features=1024, out_features=n_class, bias=True)
    model.to(device)
    set_parameter_requires_grad(model, feature_extract)
    return model


        
