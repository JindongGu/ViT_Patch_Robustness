
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import random
import math
import os
import copy
import cv2
import numpy as np


def get_mask(input_size=224, num_batch=196):
    num_row = num_col = int(math.sqrt(num_batch))
    patch_size = int(input_size / num_row)
    mask = torch.cuda.FloatTensor(num_batch, num_batch).fill_(0)
    mask[np.arange(num_batch), np.arange(num_batch)] = 1
    mask = mask.view(num_batch, num_row, num_col, 1, 1).repeat(1, 1, 1, patch_size, patch_size).permute(0, 2, 3, 1, 4).reshape(num_batch, 1, input_size, input_size)
    return mask


def preprocess_map(gradient_attns, indices=[], ori_cls=False, patch_ind=None, patch_size=32, num_patch=49):
    
    if ori_cls: return gradient_attns

    n_row = int(math.sqrt(num_patch))

    c = int(patch_ind / n_row)*patch_size -1
    r = int(patch_ind % n_row)*patch_size -1
    if c==-1: c=0
    if r==-1: r=0
    
    for i in range(len(gradient_attns)):
        if i in indices:
            gradient_attns[i, 0, :, 0] = 1
            gradient_attns[i, 0, :, -1] = 1
            gradient_attns[i, 0, 0, :] = 1
            gradient_attns[i, 0, -1, :] = 1

        # draw boxes
        gradient_attns[i, 0, r, c:c+patch_size] = 0
        gradient_attns[i, 1, r, c:c+patch_size] = 1
        gradient_attns[i, 2, r, c:c+patch_size] = 1
        
        gradient_attns[i, 0, r+patch_size, c:c+patch_size] = 0
        gradient_attns[i, 1, r+patch_size, c:c+patch_size] = 1
        gradient_attns[i, 2, r+patch_size, c:c+patch_size] = 1
        
        gradient_attns[i, 0, r:r+patch_size, c] = 0
        gradient_attns[i, 1, r:r+patch_size, c] = 1
        gradient_attns[i, 2, r:r+patch_size, c] = 1
        
        gradient_attns[i, 0, r:r+patch_size, c+patch_size] = 0
        gradient_attns[i, 1, r:r+patch_size, c+patch_size] = 1
        gradient_attns[i, 2, r:r+patch_size, c+patch_size] = 1
        
    return gradient_attns


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def show_mask_on_image(img, mask):
    cam = mask.reshape(1, 224, 224) * img
    cam = cam / np.max(cam)
    return cam


def display_img(img, mask):
    np_img = img.cpu().data.numpy()
    mask = cv2.resize(mask, (np_img.shape[2], np_img.shape[3]))
    mask = show_mask_on_image(np_img, mask)
    return mask


# get image attacked successfully
def rollout_vis(args, model, imgs, noises_list, criterion=nn.CrossEntropyLoss()):
    imgs.requires_grad = True
    cls_output = model(imgs)
    cls_y = cls_output.max(1)[1]
    mask_list = get_mask(num_batch=args.num_patch)
    torch_imgs = [] 
    
    if 'deit_tiny' in args.model: 
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
        mask = attention_rollout(imgs)
    else:
        mask = model.model.forward_features(imgs)[0].mean(dim=0).cpu().data.numpy()

    
    mask = torch.from_numpy(display_img(imgs, mask))
    torch_imgs += imgs.cpu()
    torch_imgs += mask
    
    if args.num_patch==4: interval = 1; args.patch_size=112
    if args.num_patch==49: interval = 1; args.patch_size=32
    if args.num_patch==196: interval = 50; args.patch_size=16
    for i in range(args.num_patch):
        noises = noises_list[:, i].cuda()

        mask_box = mask_list[i:i+1].cuda()
        if "patch_attack" == args.pert_type:
            adv_images = (imgs*(1-mask_box) + noises*mask_box).detach()
        elif "patch_corrupt" == args.pert_type:
            adv_images = (imgs + noises*mask_box).detach()
        adv_images.requires_grad = True
        cls_output = model(adv_images)
        cls_y = cls_output.max(1)[1]
        
        #if i%interval == 0:
        if i in [8, 25, 36]:
            if 'deit_tiny' in args.model: 
                mask = attention_rollout(adv_images)
            else:
                mask = model.model.forward_features(imgs)[0].mean(dim=0).cpu().data.numpy()
            
            mask = torch.from_numpy(display_img(adv_images, mask))
            torch_imgs += preprocess_map(adv_images.cpu(), indices=[],  patch_ind=i, patch_size=args.patch_size, num_patch=args.num_patch) 
            torch_imgs += preprocess_map(mask, indices=[],  patch_ind=i, patch_size=args.patch_size, num_patch=args.num_patch)

    return torch_imgs
        

def rollout_attn_vis(args, model, imgs, noises):
    batch_imgs = []
    images_indices = [0]
    for i in images_indices:
        batch_imgs += rollout_vis(args, model=model, imgs=imgs[i:i+1], noises_list=noises[i:i+1])

    save_image(torch.stack(batch_imgs).cpu(), 'imgs/Attention_vis_{}.jpg'.format(args.pert_type), nrow=10, padding=5, normalize=True, range=(0., 1.), scale_each=True, pad_value=1)
    print('The visualized attention is saved in the the folder "imgs/" !')
     




