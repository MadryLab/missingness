# Missingness Bias in Model Debugging.

This repository contains the code of our ICLR 2022 paper. 

**Missingness Bias in Model Debugging** </br>
*Saachi Jain\*, Hadi Salman\*, Eric Wong, Pengchuan Zhang, Vibhab Vineet, Sai Vemprala, Aleksander Madry*

```bibtex
  @inproceedings{jain2022missingness,
      title={Missingness Bias in Model Debugging},
      author={Saachi Jain and Hadi Salman and Eric Wong and Pengchuan Zhang and Vibhav Vineet and Sai Vemprala and Aleksander Madry},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=Te5ytkqsnl}
    } 
```

All scripts to run the experiments in the paper are in `patch_ablation/`. 

## Patch Ablation Experiments (Section 3)
To generate the patch ablation experiments, run the script `run_patch_ablation.py`. For example, to run the basic patch ablation experiments for a ResNet50:
```
python run_patch_ablation.py --arch resnet50 --out-dir OUT_DIR --methods Saliency_reverse --methods Saliency --ablation-patch-size 16 --saliency-map-pkl model_checkpoints/resnet_50_sal_map.pkl --filler-values 0 0 0 --skip-factor 2
```
and for a ViT-S:
```
python run_patch_ablation.py --arch deit_small_resnet_aug --out-dir OUT_DIR --methods Saliency_reverse --methods Saliency --ablation-patch-size 16 --saliency-map-pkl  model_checkpoints/resnet_50_sal_map.pkl --filler-values 0 0 0 --skip-factor 2 --use-missingness

```
## Generate LIME for a given image (Section 4)
Run `run_lime.py` to generate LIME explanations for a given image (using either blacking out or dropping tokens for ViTs). Example:
```
python run_lime.py --out-dir  OUTDIR --ablation-patch-size 16 --filler-values 0 0 0 --superpixel-type patches --arch resnet50 --num-perturbations 1000 --batch-size 64 

python run_lime.py --out-dir  OUTDIR --ablation-patch-size 16 --filler-values 0 0 0 --superpixel-type patches --arch deit_small_resnet_aug --num-perturbations 1000 --batch-size 64 --use-missingness
```
## Perform top-K ablation test (Section 4)

Run `run_lime_ablation.py` to perform the top K ablation test on LIME explanations. Example:
```
python run_lime_ablation.py --model-name resnet50 --out-pkl OUT_PKL.pkl --lime-pkl LIME_PKL --ablation-patch-size 16 --num-features 50 --feature-skip 2 --superpixel-type patches

python run_lime_ablation.py --model-name deit_small_resnet_aug_missingness --out-pkl OUT_PKL.pkl --lime-pkl LIME_PKL --ablation-patch-size 16 --num-features 50 --feature-skip 2 --superpixel-type patches
```

## Download our Models and Data

We further have the following files on dropbox since they were too large:

- [Superpixels](https://www.dropbox.com/s/1y2gqdt2yp685yd/slic_superpixel.npy?dl=0): The SLIC superpixel segmentation maps for experiments in Appendix C.6 and D.2.

- [ResNet-50 Saliency Map](https://www.dropbox.com/s/dwnsmso8xw03z9r/resnet_50_sal_map.pkl?dl=0): The saliency map used to determine the order of patches removed in Section 3.

- [Model Checkpoints](https://www.dropbox.com/s/httwdzvabivgm7i/model_checkpoints.zip?dl=0): The checkpoints for models used in this paper,

- [Missingness Results](https://www.dropbox.com/s/tpizqwgf9ph5d09/missingness_results.zip?dl=0): The data from the saved results of experiments in the paper.


## Have further questions?
Feel free to raise issues or contact us!

- [Saachi Jain](https://twitter.com/saachi_jain_)
- [Hadi Salman](https://twitter.com/hadisalmanX)
