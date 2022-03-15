import os
import numpy as np
import pickle as pkl
import sys

import sklearn.metrics
sys.path.append('../src')
import torch
import torchvision
import ds_utils
import argparse
from lime import generate_patch_grid, vit_get_patch_mask
import json
from patch_ablation_utils import reduce_mask


parser = argparse.ArgumentParser(description='Run Patch Ablation')
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--lime-pkl', type=str)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--num-features', type=int, default=190)
parser.add_argument('--feature-skip', type=int, default=5)
parser.add_argument('--ablation-patch-size', type=int, default=16)
parser.add_argument('--superpixel-type', type=str, required=True)
parser.add_argument('--model-path', type=str)
parser.add_argument('--superpixel-pkl', type=str, required=False)


args = parser.parse_args()
model_name = args.model_name
SUPERPIXEL_TYPE = args.superpixel_type
ablation_patch_size = args.ablation_patch_size
num_patches = (224//args.ablation_patch_size)**2

root_dir = "/mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/superpixel_redo/"

with open(args.out_pkl + "_env.json", "w") as f:
    json.dump(vars(args), f)

if args.lime_pkl is None:
    # do random
    print("RANDOM LIME ORDERING")
    lime_ordering = None
    use_random = True
else:
    lime_pkl = args.lime_pkl
    with open(lime_pkl, 'rb') as f:
        lime_ordering = pkl.load(f)
    if isinstance(lime_ordering, dict):
        lime_ordering = lime_ordering['lime_ordering']
    use_random = False
        
num_features_vec = np.arange(0, args.num_features, args.feature_skip)

if args.superpixel_type == 'patches':
    grid_superpixels = generate_patch_grid(args.ablation_patch_size).unsqueeze(0)
    superpixels = None
else:
    assert args.superpixel_pkl is not None
    superpixels = np.load(args.superpixel_pkl)
    
def compute_superpixel_mask(superpixel, lime_order):
    feature_mask = torch.ones_like(superpixel) * np.inf
    for idx, i in enumerate(lime_order):
        feature_mask[superpixel == i] = idx
    return feature_mask

def extend_superpixel_mask(superpixel, lime_order, num_features):
    return torch.stack([superpixel == i for i in lime_order[:num_features]]).int()

dataset = ds_utils.ImageNetMegaDS(do_mask=True)

nets = {}
arch = args.model_name
if '_missingness' in args.model_name:
    arch = arch.split('_missingness')[0]
net = ds_utils.get_full_model(arch, args.model_path, "/mnt/nfs/home/saachij/src/CausalDomainTransfer/src/config/deit.yaml")

def get_ablation(image, superpixel, lime, num_features_vec, net, using_missingness=False, reverse=False):
    # if reverse, black out everything except the features
    num_features = num_features_vec.max() + 1
    all_inputs = [ds_utils.TRANSFORM_NORMALIZE(image)]
    feature_mask = extend_superpixel_mask(superpixel, lime, num_features)
    feature_mask = feature_mask.cuda()
    curr_mask = torch.zeros_like(feature_mask[0]).cuda()
    if using_missingness:
        input_masks = [curr_mask]
        assert reverse
    for k in range(num_features):
        new_mask = curr_mask + feature_mask[k]
        if k in num_features_vec:
            if using_missingness:
                all_inputs.append(ds_utils.TRANSFORM_NORMALIZE(image))
                input_masks.append(new_mask)
            else:
                image_ = image.clone()
                image_ = image_.permute(1, 2, 0)
                if reverse:
                    image_[new_mask==1] = 0
                else:
                    image_[new_mask==0] = 0
                image_ = image_.permute(2, 0, 1)
                all_inputs.append(ds_utils.TRANSFORM_NORMALIZE(image_))
        curr_mask = new_mask
    all_inputs = torch.stack(all_inputs)
    all_inputs = all_inputs.cuda()
    if using_missingness:
        input_masks = torch.stack(input_masks).float()
        patch_masks = reduce_mask(input_masks)
        all_out = net(all_inputs, patch_mask=patch_masks)
    else:
        all_out = net(all_inputs)
    all_out = torch.softmax(all_out, 1)
    all_out = all_out.argmax(-1).cpu().numpy()
    return all_out[0] == all_out[1:]


def get_ablation_missingness(images, limes, num_features_vec, net, reverse=False):
    target = generate_patch_grid(ablation_patch_size)
    sixteen = generate_patch_grid(16)
    target_to_sixteen = {i: torch.unique(sixteen[target == i]) for i in range(0, num_patches)}
    
    num_features = num_features_vec.max() + 1
    orig_inp = ds_utils.TRANSFORM_NORMALIZE(images).cuda()
    orig_out = net(orig_inp)
    orig_label = torch.softmax(orig_out, 1).argmax(-1)
    B = len(images)
    all_out = []
    for k in range(num_features):
        if k in num_features_vec:
            all_patch_masks = []
            for b in range(B):
                lime = limes[b]
                if reverse:
                    lime_to_take = lime[(k+1):]
                else:
                    lime_to_take = lime[:(k+1)]
                all_patch_masks.append(vit_get_patch_mask(lime_to_take, target_to_sixteen).cuda())
            all_patch_masks = torch.stack(all_patch_masks)
            feat_out = net(orig_inp, patch_mask=all_patch_masks)
            feat_out = torch.softmax(feat_out, 1).argmax(-1)
            all_out.append(feat_out == orig_label)
    all_out = torch.stack(all_out, dim=1)
    return all_out.cpu().numpy()

dataset.superpixel = superpixels
dataset.lime = lime_ordering
results, reverse_results = [], []
with torch.no_grad():
    using_missingness = 'missingness' in model_name
    if using_missingness and args.superpixel_type == 'patches':
        dl = ds_utils.get_loader(dataset, bsz=64)
        for idx, mega in enumerate(dl):
            print(idx)
            images = mega['image']
            if use_random:
                mega_lime = torch.stack([torch.randperm(num_patches) for _ in range(len(images))])
            else:
                mega_lime = mega['lime']
            results.append(get_ablation_missingness(images, mega_lime, num_features_vec, net))
            reverse_results.append(get_ablation_missingness(images, mega_lime, 
                                                      num_features_vec, net, reverse=True))
        results = np.concatenate(results)
        reverse_results = np.concatenate(reverse_results)
    else:
        dl = ds_utils.get_loader(dataset, bsz=1)
        for idx, mega in enumerate(dl):
            if idx % 100 == 0:
                print(idx)
            if use_random:
                mega_lime = torch.randperm(num_patches)
            else:
                mega_lime = mega['lime'].squeeze(0)
            if SUPERPIXEL_TYPE == 'patches':
                superpixel = grid_superpixels.squeeze(0)
            else:
                superpixel = mega['superpixel'].squeeze(0)
                mega_lime[mega_lime <= superpixel.max()]
            if not using_missingness:
                results.append(get_ablation(mega['image'][0], superpixel, 
                                            mega_lime, num_features_vec, net,
                                            using_missingness=using_missingness))
            reverse_results.append(get_ablation(mega['image'][0], superpixel, 
                                                mega_lime, num_features_vec, net,
                                                using_missingness=using_missingness, reverse=True))
        if not using_missingness:
            results = np.stack(results)
        reverse_results = np.stack(reverse_results)
        
output_result = {
    'results': results,
    'reverse_results': reverse_results,
    'num_features_vec': num_features_vec,
}
with open(args.out_pkl, 'wb') as f:
    pkl.dump(output_result, f)
