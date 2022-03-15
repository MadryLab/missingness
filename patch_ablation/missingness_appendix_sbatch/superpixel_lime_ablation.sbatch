#!/bin/bash
#SBATCH --gpus 1
#SBATCH --partition normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name deit_lime_ablation_random
#SBATCH --output=logs/deit_lime_ablation_random.err
#SBATCH --error=logs/deit_lime_ablation_random.out
#SBATCH --nodelist deep-gpu-10
set +x
cd /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation

PATCH_SIZE=16
ARCH="deit_small_resnet_aug_missingness"
ROOTDIR="/mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/superpixel_redo_slic_lime"
NUM_FEATURES=50
FEATURE_SKIP=2
ORDER="deit_small_resnet_aug_missingness"
SUPERPIXEL_PKL="/mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps/slic_superpixel.npy"

#python run_lime_ablation.py --model-name ${ARCH} --out-pkl ${ROOTDIR}/${ARCH}/ablations/${ORDER}_order.pkl --lime-pkl ${ROOTDIR}/${ORDER}/lime.pkl --ablation-patch-size ${PATCH_SIZE} --num-features ${NUM_FEATURES} --feature-skip ${FEATURE_SKIP} --superpixel-type superpixel --superpixel-pkl $SUPERPIXEL_PKL

python run_lime_ablation.py --model-name ${ARCH} --out-pkl ${ROOTDIR}/${ARCH}/ablations/random_order.pkl --ablation-patch-size ${PATCH_SIZE} --num-features ${NUM_FEATURES} --feature-skip ${FEATURE_SKIP} --superpixel-type superpixel --superpixel-pkl $SUPERPIXEL_PKL