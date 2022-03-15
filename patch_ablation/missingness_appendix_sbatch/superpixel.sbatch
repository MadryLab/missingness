#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name superpixel_gen
#SBATCH --output=logs/superpixel_gen.out
#SBATCH --error=logs/superpixel_gen.err
set +x
cd /mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation

python generate_superpixel.py --out-file "/mnt/nfs/home/saachij/src/CausalDomainTransfer/patch_ablation/missingness_results/saliency_maps/slic_superpixel.npy"