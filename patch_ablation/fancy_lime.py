import torch
import sklearn
import torch as ch
import ds_utils
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from skimage.segmentation import quickshift, slic
from sklearn.linear_model import Ridge
import time
import torch

def tv_norm(img, tv_beta):
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def get_lime_for_image(pm, dataset, net, num_steps=1000, transform_normalize=ds_utils.TRANSFORM_NORMALIZE, 
                      batch_size=64):
    total_patches = pm.total_patches.item()
    loader = ds_utils.get_loader(dataset, bsz=batch_size) #bsz 64
    all_lime, all_coeffs = [], []

    for batch_index, mega in enumerate(loader):
        if batch_index % 10 == 0:
            print(batch_index, len(loader))
        images = mega['image'].cuda()
        B = images.shape[0]
        all_logits = []
        orig_inp = transform_normalize(images)
        orig_out = net(orig_inp)
        orig_pred = ch.softmax(orig_out, 1).argmax(-1).cpu()

        mask = torch.ones(B, 14, 14).cuda()
        mask_var = torch.autograd.Variable(mask, requires_grad=True)
        optimizer = torch.optim.Adam([mask_var], lr=0.1)
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
        black_vals = torch.zeros((B, 3, 224, 224)).cuda()

        images_norm = transform_normalize(images)
        black_vals_norm = transform_normalize(black_vals)
        for t in range(num_steps):
            upsampled = upsample(mask_var.unsqueeze(1))
            inp = images_norm * upsampled + black_vals_norm * (1-upsampled)
            inp += (torch.randn((B, 3, 224, 224))*0.2).cuda()
            out = ch.softmax(net(inp), 1)
            loss = 0
            for b in range(B):
                loss += 0.01*torch.mean(torch.abs(1-mask_var[b])) + 0.2*tv_norm(mask_var[b], 3) + out[b][orig_pred[b]]
            if t % 10 == 0:
                print(t, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mask_var.data.clamp_(0, 1)
        mask_var = mask_var.reshape(B, -1)
        all_lime.append(mask_var.argsort(dim=1, descending=False).detach().cpu().numpy())
        all_coeffs.append(mask_var.detach().cpu().numpy())
    return np.concatenate(all_lime), np.concatenate(all_coeffs)
