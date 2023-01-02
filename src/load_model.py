# -*- coding: <encoding-name> -*-
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
from torchvision import transforms


class WrapModel(nn.Module):
        # wrap the model so that the input range is in [0, 1]
	def __init__(self, model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		super(WrapModel, self).__init__()

		self.model = model
		self.mean = torch.as_tensor(mean).view(1, 3, 1, 1)
		self.std = torch.as_tensor(std).view(1, 3, 1, 1)

	def forward(self, x):
		device = x.get_device()
		self.mean = self.mean.to(device)
		self.std = self.std.to(device)
		x = (x - self.mean.detach()) /self.std.detach()
		return self.model(x)
       

def load_model(args):

    model = timm.create_model(args.model, pretrained=True).eval()
    
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    config = resolve_data_config({}, model=model)
    data_transforms = create_transform(**config)

    pre_transforms = data_transforms.transforms
    Norm = pre_transforms[-1]
    
    data_transforms.transforms = data_transforms.transforms[0:-1]
    return WrapModel(model, mean=Norm.mean, std=Norm.std).cuda(), data_transforms







