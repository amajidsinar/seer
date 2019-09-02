import torch
from nf_trainer.trainer import set_parameter_requires_grad
from torch import nn
from nf_trainer.utils import create_instance
import importlib

__all__ = ['MobileNet_v3']

def MobileNet_v3(use_pretrained, feature_extract, n_class, gpu_id, mode):

    model = importlib.import_module('nf_trainer.architecture.mobilenet_v3')
    model = getattr(model, "MobileNetV3")(mode=mode)
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    if use_pretrained:
        assert mode == 'small', "Pretrained model only available in mode = 'small'"
        state_dict = torch.load('nodeflux/pretrained_weights/mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        
    model.classifier[1] = nn.Linear(1280, n_class, bias=True)
    model.to(device)
    set_parameter_requires_grad(model, feature_extract)
    return model





        
