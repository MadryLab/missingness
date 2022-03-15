#!/bin/bash
#SBATCH --gpus 1
#SBATCH --partition normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name superpixel_lime_deit
#SBATCH --output=logs/superpixel_lime_deit.out
#SBATCH --error=logs/superpixel_lime_deit.err
#SBATCH --nodelist deep-gpu-10
set +x
cd /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation

FOLDER_OUT="missingness_results/superpixel_redo_slic_lime/"


#mkdir $FOLDER_OUT/resnet50
#python run_lime.py --out-dir  $FOLDER_OUT/resnet50 --ablation-patch-size 16 --filler-values 0 0 0 --superpixel-type superpixel --arch resnet50 --num-perturbations 1000 --batch-size 64 --superpixel-pkl /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps/slic_superpixel.npy 

mkdir $FOLDER_OUT/deit_small_resnet_aug_missingness
python run_lime.py --out-dir  $FOLDER_OUT/deit_small_resnet_aug_missingness --ablation-patch-size 16 --filler-values 0 0 0 --superpixel-type superpixel --arch deit_small_resnet_aug --num-perturbations 1000 --batch-size 64 --superpixel-pkl /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps/slic_superpixel.npy --use-missingness