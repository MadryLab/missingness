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
parser.add_argument('--saliency-map-pkl', default=None, type=str)
parser.add_argument('--methods', type=str, action='append')
parser.add_argument('--model-path', type=str, default=None)
parser.add_argument('--ablation-patch-size', default=16, type=int)
parser.add_argument('--filler-values', type=float, nargs=3, default=[0.0,0.0,0.0])
parser.add_argument('--superpixel', action='store_true')
parser.add_argument('--skip-random', action='store_true')
parser.add_argument('--superpixel-pkl', default=None, type=str)
parser.add_argument('--image-net-cache-file', default=None, type=str)
parser.add_argument('--skip-factor', default=1, type=int)
parser.add_argument('--superpixel-type', type=str, default='patches')
parser.add_argument('--use-missingness', action='store_true')



args = parser.parse_args()
print(args)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with open(os.path.join(args.out_dir, "env.json"), "w") as f:
    json.dump(vars(args), f)
    
dataset = ds_utils.CIFARMegaDS(cache_file=args.image_net_cache_file)
    
methods = args.methods
if methods is None:
    methods = []
ARCH = args.arch
net = ds_utils.get_cifar_model(args.arch)

# LOAD SUPERPIXEL
if args.superpixel:
    assert args.superpixel_pkl is not None
    superpixels = np.load(args.superpixel_pkl)
    dataset.superpixel = superpixels
    pm = SuperPixelPatchMaster(ARCH, img_size=224, total_patches=superpixels.max()+1)
else:
    pm = PatchMaster(ARCH, patch_size=args.ablation_patch_size)


runner = BaseExperimentRunner(patchmaster=pm, dataset=dataset, net=net)

methods_to_run = methods if args.skip_random else ['random'] + methods

# LOAD SALIENCY MAP
if args.saliency_map_pkl is None:
    all_saliency_maps = {}
    interp_methods = set([m.split("_reverse")[0] for m in methods])
    for m in interp_methods:
        print(f'Method: {m}')
        sal_maps = []
        dl = ds_utils.get_loader(dataset, bsz=sal_utils.BATCH_SIZES[m])
        for i, mega in enumerate(dl):
            attr = runner.perform_interp(mega, m)
            sal_maps.append(attr)
        all_saliency_maps[m] = torch.cat(sal_maps, 0)
    sal_map_pkl_file = os.path.join(args.out_dir, 'sal_map.pkl')
    with open(sal_map_pkl_file, 'wb') as f:
        pkl.dump(all_saliency_maps, f)
else:
    with open(args.saliency_map_pkl, 'rb') as f:
        all_saliency_maps = pkl.load(f)
dataset.saliency = all_saliency_maps

# RUN ENVELOPE
if args.use_missingness:
    use_masking = False
else:
    use_masking = True
if args.superpixel:
    mask_accs, mask_soft_labels, mask_preds, mask_debugs = runner.get_envelope(methods_to_run,
                                                                              all_saliency_maps, 
                                                                              use_masking=use_masking,
                                                                              filler_value=args.filler_values,
                                                                              skip_factor=args.skip_factor,
                                                                              num_features=50,
                                                                              use_superpixel=True,
                                                                              transform_normalize=ds_utils.CIFAR_NORMALIZE)
else:
    mask_accs, mask_soft_labels, mask_preds, mask_debugs = runner.get_envelope(methods_to_run,
                                                              all_saliency_maps, 
                                                              use_masking=use_masking,
                                                              filler_value=args.filler_values,
                                                              skip_factor=args.skip_factor,
                                                              transform_normalize=ds_utils.CIFAR_NORMALIZE)
out_file = os.path.join(args.out_dir, 'envelope.pkl')
with open(out_file, 'wb') as f:
    pkl.dump({
        'mask_accs': mask_accs,
        'mask_soft_labels': mask_soft_labels,
        'mask_preds': mask_preds,
        'mask_debugs': mask_debugs,
    }, f)
