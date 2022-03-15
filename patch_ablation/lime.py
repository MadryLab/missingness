import torch
import sklearn
import torch as ch
import ds_utils
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from skimage.segmentation import quickshift, slic
from sklearn.linear_model import Ridge
import time

def generate_patch_grid(patch_size):
    num_patches = 224//patch_size
    patch_mask = torch.arange(num_patches**2).reshape(num_patches, num_patches).unsqueeze(0).unsqueeze(0).float()
    patch_mask = torch.nn.Upsample(scale_factor=patch_size)(patch_mask).long()
    patch_mask = patch_mask.squeeze(0).squeeze(0)
    return patch_mask

def get_superpixel_for_image(dataset, superpixel_type, patch_size=16):
    out = []
    loader = ds_utils.get_loader(dataset, bsz=1)
    if superpixel_type == 'patches':
        assert patch_size % 16 == 0
        patch_mask = generate_patch_grid(patch_size)
    for batch_index, mega in enumerate(loader):
        if batch_index % 10 == 0:
            print(batch_index, len(loader))
        img_tensor = mega['image'].squeeze(0)
        if superpixel_type == 'patches':
            out.append(patch_mask)
        elif superpixel_type == 'quickshift':
            out.append(torch.tensor(
                quickshift(
                    img_tensor.permute(1, 2, 0).double().numpy(),
                    kernel_size=4,
                    max_dist=200, 
                    ratio=0.2)
            ).long())
        elif superpixel_type == 'slic':
            out.append(torch.tensor(
                slic(img_tensor.permute(1, 2, 0).double().numpy(),
                     n_segments=150,
                     compactness=10,
                     sigma=1,
                     start_label=0)).long())
        else:
            raise ValueError()
    return torch.stack(out)

def vit_get_patch_mask(target_patches_to_keep, target_to_sixteen): #target_patches_to_keep should be K dimensional
    sixteens = torch.cat([target_to_sixteen[t.item()] for t in target_patches_to_keep])
    sixteens = sixteens + 1
    sixteens = torch.cat([torch.tensor([0]), sixteens])
    return sixteens

def fit_lime_model(distance, patches_to_keep, logits):
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distance**2)/kernel_width**2)) #Kernel function
    simpler_model = Ridge(alpha=1, fit_intercept=True)
    simpler_model.fit(X=patches_to_keep, y=logits , sample_weight=weights)
    coeff_order = np.argsort(simpler_model.coef_)[::-1]
    return coeff_order, simpler_model.coef_

def get_lime_for_image_missingness(pm, dataset, net, num_perturbations=1000, 
                                   ablation_patch_size=16, transform_normalize=ds_utils.TRANSFORM_NORMALIZE,
                                  batch_size=64):
    total_patches = pm.total_patches.item()
    target = generate_patch_grid(ablation_patch_size)
    sixteen = generate_patch_grid(16)
    target_to_sixteen = {i: torch.unique(sixteen[target == i]) for i in range(0, pm.total_patches)}
    
    patches_to_keep = np.random.binomial(1, 0.5, size=(num_perturbations, total_patches))
    bool_patches_to_keep = patches_to_keep > 0
    orig_img = np.ones((1, total_patches))
    distances = pairwise_distances(patches_to_keep, orig_img, metric='cosine').ravel()     
    all_patch_keepers = []
    for t in range(num_perturbations):
        patch_keepers = torch.arange(total_patches)[bool_patches_to_keep[t]]
        patch_keepers = vit_get_patch_mask(patch_keepers, target_to_sixteen)
        all_patch_keepers.append(patch_keepers)
    
    loader = ds_utils.get_loader(dataset, bsz=batch_size)
    all_lime, all_coeffs = [], []
    with torch.no_grad():
        for batch_index, mega in enumerate(loader):
            if batch_index % 10 == 0:
                print(batch_index, len(loader))
            images = mega['image']
            B = images.shape[0]
            all_logits = []
            orig_inp = transform_normalize(images).cuda()
            orig_out = net(orig_inp)
            orig_pred = ch.softmax(orig_out, 1).argmax(-1).cpu().numpy()
            for t in range(num_perturbations):
                if t % 10 == 0:
                    print(t)
                patch_keepers = all_patch_keepers[t].unsqueeze(0).expand(B, -1).cuda()
                out = net(orig_inp, patch_mask=patch_keepers)
                out = ch.softmax(out, 1)
                all_logits.append(out.cpu())
            all_logits = torch.stack(all_logits, dim=1)
            for b in range(B):
                coeff_order, coeff = fit_lime_model(distances, patches_to_keep, all_logits[b, :,orig_pred[b]])
                all_lime.append(coeff_order)
                all_coeffs.append(coeff)
    return np.stack(all_lime), np.stack(all_coeffs)



def get_lime_for_image(pm, dataset, net, num_perturbations=1000, filler_value=(0,0,0), transform_normalize=ds_utils.TRANSFORM_NORMALIZE, use_generated_patches=False, ablation_patch_size=16,
                      batch_size=64, use_slow_missingness=False):
    total_patches = pm.total_patches.item()
    loader = ds_utils.get_loader(dataset, bsz=batch_size) #bsz 64
    all_lime, all_coeffs = [], []
    patches_to_keep = np.random.binomial(1, 0.5, size=(num_perturbations, total_patches))
    patches_to_exclude = (1-patches_to_keep).astype(np.bool)
    
    if use_generated_patches:
        grid_superpixels = generate_patch_grid(ablation_patch_size).unsqueeze(0)
        grid_superpixels = ds_utils.extend_superpixel(grid_superpixels, total_patches=total_patches)
        grid_distances = pairwise_distances(patches_to_keep, np.ones((1, total_patches)), metric='cosine').ravel() 

    with torch.no_grad():
        for batch_index, mega in enumerate(loader):
            if batch_index % 10 == 0:
                print(batch_index, len(loader))
            images = mega['image']
            B = images.shape[0]
            if use_generated_patches:
                assert len(grid_superpixels.shape) == 4
                superpixel_extension = grid_superpixels.expand(B, -1, -1, -1).cuda()
                distances = [grid_distances for _ in range(B)]
            else:
                superpixels = mega['superpixel']
                superpixel_extension = ds_utils.extend_superpixel(superpixels, total_patches=total_patches).cuda()
                distances = []
                for b in range(B):
                    orig_img = np.zeros((1, total_patches))
                    orig_img[:, :superpixels[b].max()+1] = 1
                    distances.append(pairwise_distances(patches_to_keep, orig_img, metric='cosine').ravel())
                
            all_logits = []
            orig_inp = transform_normalize(images).cuda()
            orig_out = net(orig_inp)
            orig_pred = ch.softmax(orig_out, 1).argmax(-1).cpu().numpy()
            for t in range(num_perturbations):
                if t % 10 == 0:
                    print(t)
                patches_to_mask = torch.arange(total_patches)[patches_to_exclude[t]]
                if use_slow_missingness:
                    deit_patch_masks = pm.slow_mask_orders_single_image(images, 
                                                                         patches_to_mask, 
                                                                         filler_value, 
                                                                         superpixel_extension)
                    out = net(orig_inp, patch_mask=deit_patch_masks)
                else:
                    inp = pm.batch_single_mask_image(images, patches_to_mask, filler_value, superpixel_extension)
                    inp = transform_normalize(inp).cuda()
                    out = net(inp)
                out = ch.softmax(out, 1)
                all_logits.append(out.cpu())
            all_logits = torch.stack(all_logits, dim=1)
            for b in range(B):
                coeff_order, coeff = fit_lime_model(distances[b], patches_to_keep, all_logits[b, :,orig_pred[b]])
                all_lime.append(coeff_order)
                all_coeffs.append(coeff)
    return np.stack(all_lime), np.stack(all_coeffs)
