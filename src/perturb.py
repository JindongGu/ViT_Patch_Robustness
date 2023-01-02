
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from imagenet_c import corrupt


class Mask_Value:
    def __init__(self, input_size=224, num_batch=196):
        num_row = num_col = int(math.sqrt(num_batch))
        patch_size = int(input_size / num_row)
        mask = torch.cuda.FloatTensor(num_batch, num_batch).fill_(0)
        mask[np.arange(num_batch), np.arange(num_batch)] = 1
        self.mask = mask.view(num_batch, num_row, num_col, 1, 1).repeat(1, 1, 1, patch_size, patch_size).permute(0, 2, 3, 1, 4).reshape(num_batch, 1, input_size, input_size)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def patch_attack(args, Mask, model, X, y, eps, nb_iter, eps_iter, lower_limit=0.0, upper_limit=1.0, input_size=224, num_batch=196, save_noise=True):
    num_row = num_col = int(math.sqrt(num_batch))
    
    mask_batch = Mask.mask
    X_batch = X.repeat(num_batch, 1, 1, 1)
    delta_batch = torch.rand(X_batch.shape).cuda()
    delta_save = torch.zeros(X_batch.shape).cuda()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    bn_size = num_batch
    indices_batch = np.arange(num_batch)
    for ind in range(int(len(X_batch)/bn_size)):
        indices = indices_batch[:bn_size]
        X = X_batch[ind*bn_size:(ind+1)*bn_size]
        delta = delta_batch[ind*bn_size:(ind+1)*bn_size]
        delta.requires_grad = True
        mask = mask_batch[ind*bn_size:(ind+1)*bn_size]

        # for replacement
        X = X*(1-mask)

        for iterd in range(nb_iter):
            if len(indices) == 0: break
            output = model(X[indices] + delta[indices]*mask[indices])
            succe = (output.max(1)[1] != y).cpu().data.numpy()
            
            succe_indices = np.where(succe)[0].tolist()
            if save_noise and (len(succe_indices) > 0):
                delta_save[ind*bn_size + indices[succe_indices]] += delta[indices[succe_indices]]
            
            loss = criterion(output, y.expand(len(output)))
            loss.backward()
            grad = delta.grad
            delta.data = torch.clamp(delta + eps_iter*torch.sign(grad), lower_limit, upper_limit)
            delta.grad.zero_()
            model.zero_grad()
           
            indices = np.delete(indices, np.where(succe)[0].tolist())
            if iterd%100==0: print('{}/{} batch, {} attack iteration, {}/{} images are still correct!'.format(ind+1, int(len(X_batch)/bn_size), iterd, len(indices), num_batch)); 

    return delta_save[None, :, :, :, :]
            

def patch_corrupt(args, X_batch, Mask, input_size, num_patch):
    bs = X_batch.shape[0]
    corrupt_outs = []
    for i in range(len(X_batch)):
        X = (X_batch[i].cpu().data.numpy().transpose((1,2,0))*255).astype(np.uint8)
        corrupt_outs.append(corrupt(X, severity=args.severity, corruption_number=args.corruption_number,input_size=args.input_size))
    corrupt_outs = np.stack(corrupt_outs, axis=0).transpose((0,3,1,2))
    perturb = torch.from_numpy(corrupt_outs/255.).to(args.device).float() - X_batch

    perturb = perturb.view(bs, 1, 3, input_size, input_size) * Mask.mask
    perts_inp = X_batch.view(bs, 1, 3, input_size, input_size).repeat(1, num_patch, 1, 1, 1) + perturb
    perts_inp = perts_inp.view(bs*num_patch, 3, input_size, input_size)
    return perturb
            

def patch_perturb(args, model, X_batch, y_batch, Mask=None, topN=1):
    num_patch = args.num_patch
    num_row = num_col = int(math.sqrt(num_patch))
    input_size = args.input_size
    patch_size = int(input_size / num_row)

    X_batch = X_batch.to(args.device)
    y_batch = y_batch.to(args.device)
    bs = len(X_batch)

    Mask = Mask_Value(num_batch=num_patch)

    if "patch_attack" == args.pert_type:
        perts = patch_attack(args, Mask, model, X_batch, y_batch, eps=args.epsilon, nb_iter=args.attack_iters, eps_iter=args.alpha, input_size=args.input_size, num_batch=args.num_patch)

    elif "patch_corrupt" == args.pert_type:
        perts = patch_corrupt(args, X_batch, Mask, input_size, num_patch)

    return perts



