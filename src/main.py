
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.io import read_image

import argparse
import pickle
import random
import cv2
import os
import copy
import numpy as np
import timm
import time

from load_model import load_model, WrapModel
from perturb import patch_perturb
from rollout_vis import  rollout_attn_vis


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default= 'deit_tiny_patch16_224', type=str)
    parser.add_argument('--dataset', default= 'imagenet', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)

    # for patch attack
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_patch', default=49, type=int)
    parser.add_argument('--patch_size', default=32, type=int)

    # adversarial attacks
    parser.add_argument('--pert_type', default='patch_corrupt', type=str, help='patch_corrupt, patch_attack')
    parser.add_argument('--norm', default='Linf', type=str, help='L0, L0+Linf, L0+sigma')
    parser.add_argument('--attack_iters', default=10000, type=int, help='Attack iterations')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--delta_init', default='random', choices=['zero', 'random'], help='Perturbation initialization method')
    
    # natural corruption
    parser.add_argument('--severity', default=5, type=int)
    parser.add_argument('--corruption_number', default=0, type=int)
    parser.add_argument('--num_patch_perb', default=1, type=int)

    # attention visualization
    parser.add_argument('--head_fusion', type=str, default='mean')
    parser.add_argument('--discard_ratio', type=float, default=0.9)

    parser.add_argument('--device', default='cpu', type=str, help='the current device')
    return parser.parse_args()



def eval(args, model, image_path):
    # preprocess img
    img = read_image(image_path).float()
    data_transforms = torch.nn.Sequential(
            transforms.Resize(size=248, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224))
            )
    X = data_transforms(img)[None, :, :, :]
    X = X / 255.
    X = X.to(args.device)

    # get prediction/gt class
    y = model(X).max(1)[1].detach()

    # get and save patch perturbation
    perts = patch_perturb(args, model, X, y)
    
    # visualize attention of ViT on the perturbated img
    rollout_attn_vis(args, model, X, perts)



if __name__ == '__main__':
    
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # specify attack hyperparameter
    args.epsilon /= 255
    args.alpha /= 255

    # load model
    model, _ = load_model(args) 
    model.eval()

    # load data
    image_path = 'imgs/dog.jpg'
    
    # evaluate the robustness of the specified model
    eval(args, model, image_path)



