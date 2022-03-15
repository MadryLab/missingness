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
from lime import get_lime_for_image, get_superpixel_for_image, get_lime_for_image_missingness
import json
import numpy as np

parser = argparse.ArgumentParser(description='Run Patch Ablation')
parser.add_argument('--arch', required=True, type=str)
parser.add_argument('--out-dir', required=True, type=str)
parser.add_argument('--ablation-patch-size', default=16, type=int)
parser.add_argument('--filler-values', type=float, nargs=3, default=[0.0,0.0,0.0])
parser.add_argument('--superpixel-pkl', type=str, required=False)
parser.add_argument('--superpixel-type', type=str, default='patches')
parser.add_argument('--use-missingness', action='store_true')
parser.add_argument('--num-perturbations', type=int, default=1000)
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
if args.superpixel_type != 'patches':
    assert args.superpixel_pkl is not None
    superpixels = np.load(args.superpixel_pkl)
    dataset.superpixel = superpixels
    total_patches = superpixels.max()+1
else:
    total_patches = torch.tensor((224//args.ablation_patch_size)**2)
pm = SuperPixelPatchMaster(ARCH, img_size=224, total_patches=total_patches)
if args.use_missingness and args.superpixel_type == 'patches':
    lime_ordering, lime_coeffs = get_lime_for_image_missingness(pm=pm, dataset=dataset, net=net, 
                                                                num_perturbations=args.num_perturbations,
                                                                ablation_patch_size=args.ablation_patch_size,
                                                                batch_size=args.batch_size)
else:
    use_generated_patches = args.superpixel_type == 'patches'
    lime_ordering, lime_coeffs = get_lime_for_image(pm=pm, dataset=dataset, net=net, 
                                                    num_perturbations=args.num_perturbations, 
                                                    filler_value=args.filler_values,
                                                    use_generated_patches=use_generated_patches,
                                                    ablation_patch_size=args.ablation_patch_size,
                                                    batch_size=args.batch_size,
                                                    use_slow_missingness=args.use_missingness)
with open(os.path.join(args.out_dir, 'lime.pkl'), 'wb') as f:
    pkl.dump({'lime_ordering': lime_ordering, 'lime_coeffs': lime_coeffs}, f)
        
