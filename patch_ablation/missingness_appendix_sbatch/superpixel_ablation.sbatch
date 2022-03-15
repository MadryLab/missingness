#!/bin/bash
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name superpixel_deit
#SBATCH --output=logs/superpixel_deit.out
#SBATCH --error=logs/superpixel_deit.err
#SBATCH --nodelist deep-gpu-10
set +x
cd /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation

SKIP_FACTOR=1
ABLATION_PATCH_SIZE=16 
FIRST_COLOR=0 
SECOND_COLOR=0
THIRD_COLOR=0

FOLDER_NAME="slic_superpixel"
SAL_MAP_FOLDER="/mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps"

mkdir missingness_results/$FOLDER_NAME

#python run_patch_ablation.py --arch resnet50 --out-dir missingness_results/$FOLDER_NAME/resnet50 --methods Saliency_reverse --methods Saliency --ablation-patch-size $ABLATION_PATCH_SIZE --saliency-map-pkl $SAL_MAP_FOLDER/resnet_50_sal_map.pkl --filler-values $FIRST_COLOR $SECOND_COLOR $THIRD_COLOR --skip-factor $SKIP_FACTOR --superpixel --superpixel-pkl /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps/slic_superpixel.npy

python run_patch_ablation.py --arch deit_small_resnet_aug --out-dir missingness_results/$FOLDER_NAME/deit_small_resnet_aug --methods Saliency_reverse --methods Saliency --ablation-patch-size $ABLATION_PATCH_SIZE --saliency-map-pkl $SAL_MAP_FOLDER/resnet_50_sal_map.pkl --filler-values $FIRST_COLOR $SECOND_COLOR $THIRD_COLOR --skip-factor $SKIP_FACTOR --superpixel --superpixel-pkl /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps/slic_superpixel.npy --use-missingness
