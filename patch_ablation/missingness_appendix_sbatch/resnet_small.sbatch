#!/bin/bash
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name resnet_small
#SBATCH --output=logs/resnet_small.out
#SBATCH --error=logs/resnet_small.err
#SBATCH --nodelist deep-gpu-10
set +x
cd /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation

SKIP_FACTOR=2
ABLATION_PATCH_SIZE=16 # skip_factor 2: 14, 16, skip_factor 1: 28, 32, 56, 64
FIRST_COLOR=0 #-2 for random pixel, -1 for random patch, [0.485, 0.456, 0.406] for mean, -3 for blur
SECOND_COLOR=0
THIRD_COLOR=0

FOLDER_NAME="patch_${ABLATION_PATCH_SIZE}_${FIRST_COLOR}_${SECOND_COLOR}_${THIRD_COLOR}"
SAL_MAP_FOLDER="/mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps"

mkdir missingness_results/$FOLDER_NAME

python run_patch_ablation.py --arch resnet18 --out-dir missingness_results/$FOLDER_NAME/resnet18 --methods Saliency_reverse --methods Saliency --ablation-patch-size $ABLATION_PATCH_SIZE --saliency-map-pkl $SAL_MAP_FOLDER/resnet_50_sal_map.pkl --filler-values $FIRST_COLOR $SECOND_COLOR $THIRD_COLOR --skip-factor $SKIP_FACTOR
