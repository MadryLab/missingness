import os
import sys
import torch
from patch_ablation_utils import PatchMaster, SuperPixelPatchMaster
sys.path.append('../src')
import ds_utils
import saliency_utils as sal_utils
from base_experiments import BaseExperimentRunner
import time
import argparse
import pickle as pkl
from fancy_lime import get_lime_for_image
import json
import numpy as np

parser = argparse.ArgumentParser(description='Run Patch Ablation')
parser.add_argument('--arch', required=True, type=str)
parser.add_argument('--out-dir', required=True, type=str)
parser.add_argument('--num-steps', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--model-path', type=str)


args = parser.parse_args()
print(args)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with open(os.path.join(args.out_dir, "env.json"), "w") as f:
    json.dump(vars(args), f)
    
dataset = ds_utils.ImageNetMegaDS()
ARCH = args.arch
net = ds_utils.get_full_model(args.arch, args.model_path)

total_patches = torch.tensor((224//16)**2)
pm = SuperPixelPatchMaster(ARCH, img_size=224, total_patches=total_patches)
lime_ordering, lime_coeffs = get_lime_for_image(pm=pm, dataset=dataset, net=net, 
                                                num_steps=args.num_steps, 
                                                batch_size=args.batch_size)
with open(os.path.join(args.out_dir, 'lime.pkl'), 'wb') as f:
    pkl.dump({'lime_ordering': lime_ordering, 'lime_coeffs': lime_coeffs}, f)
        
