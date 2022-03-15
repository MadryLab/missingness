import os
import sys
import torch
from patch_ablation_utils import PatchMaster, SuperPixelPatchMaster
sys.path.append('../src')
import ds_utils
import saliency_utils as sal_utils
import time
import argparse
import pickle as pkl
from lime import get_superpixel_for_image
import json
import numpy as np
parser = argparse.ArgumentParser(description='Run Patch Ablation')

parser.add_argument('--out-file', required=True, type=str)
args = parser.parse_args()

dataset = ds_utils.ImageNetMegaDS()

superpixels = get_superpixel_for_image(dataset, 'slic').numpy()
np.save(args.out_file, superpixels)
