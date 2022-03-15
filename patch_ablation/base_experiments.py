import torch
import torch as ch
import numpy as np
import ds_utils as ds_utils
import saliency_utils as sal_utils
import time
class BaseExperimentRunner:
    def __init__(self, patchmaster, dataset, net):
        self.dataset = dataset
        self.pm = patchmaster
        self.net = net
    
    def perform_interp(self, mega, m):
        x = mega['image']
        y = mega['labels']
        inp = ds_utils.TRANSFORM_NORMALIZE(x)
        inp = inp.cuda()
        out = self.net(inp)
        out = ch.softmax(out, 1)
        score, pred = out.max(-1)
        attr_return = sal_utils.perform_saliency_batch_size(patch_master=self.pm, 
                                                            net=self.net, m=m, 
                                                            inp=inp, pred=pred, 
                                                            orig_inp=x)
        return attr_return
            
    
    def get_patch_orders(self, m, x, mega):
        reverse_orders = '_reverse' in m
        if m == 'random':
            patch_orders = self.pm.generate_random_patch_orders(x.shape[0])
        else:
            tag = m.split('_reverse')[0]
            S = mega[f"saliency_{tag}"]
            flattened_sal_map, patch_orders = self.pm.get_patch_ordering(S)
        if reverse_orders:
            if torch.is_tensor(patch_orders):
                patch_orders = patch_orders.numpy()
            patch_orders = patch_orders[:, ::-1].copy()
            patch_orders = torch.tensor(patch_orders)
        return patch_orders
    
    def get_superpixel_patch_orders(self, m, x, mega, num_features, superpixels):
        reverse_orders = '_reverse' in m
        if m == 'random':
            patch_orders = self.pm.generate_random_patch_orders(superpixels=superpixels, num_features=num_features)
        else:
            tag = m.split('_reverse')[0]
            S = mega[f"saliency_{tag}"]
            if reverse_orders:
                patch_orders = self.pm.get_superpixel_patch_ordering(S=S,
                                                      num_features=num_features,
                                                      superpixels=superpixels,
                                                      reverse=True)
            else:
                patch_orders = self.pm.get_superpixel_patch_ordering(S=S,
                                                  num_features=num_features,
                                                  superpixels=superpixels,
                                                  reverse=False)
        return patch_orders
                        
    
    def get_envelope(self, methods, all_saliency_maps, use_masking=True,
                     filler_value=(0,0,0), batch_size=500, skip_factor=1,
                     num_features=None, use_superpixel=False, transform_normalize=ds_utils.TRANSFORM_NORMALIZE):
        dl = ds_utils.get_loader(self.dataset, bsz=batch_size)
        all_accs, all_soft_labels, all_preds, all_debugs = {}, {}, {}, {}
        if num_features is None:
            num_features = self.pm.total_patches
        for m in methods:
            with torch.no_grad():
                accs, soft_labels, preds, debugs = [], [], [], []
                for i, mega in enumerate(dl):
                    x, y =  mega['image'], mega['labels']
                    x = x.cuda()
                    print("batch", i, "method", m)
                    orig_inp = transform_normalize(x).cuda()
                    if use_superpixel:
                        superpixels = mega['superpixel']
                        superpixel_extension = ds_utils.extend_superpixel(superpixels, total_patches=self.pm.total_patches)
                        patch_orders = self.get_superpixel_patch_orders(m=m, x=x, mega=mega, 
                                                                        num_features=num_features,
                                                                        superpixels=superpixels)
                    else:
                        superpixel_extension = None
                        patch_orders = self.get_patch_orders(m=m, x=x, mega=mega)
                    mini_accs, mini_soft_labels, mini_preds, mini_debugs = [], [], [], []
                    for nums_to_exclude in np.arange(0, num_features + 1, skip_factor):
                        if nums_to_exclude % 20 == 0:
                            print(nums_to_exclude)
                        if use_masking:
                            if nums_to_exclude == 0:
                                inp = x
                            else:
                                inp = self.pm.mask_images(patch_orders, nums_to_exclude, x,
                                                          filler_value=filler_value,
                                                          superpixel_extension=superpixel_extension)
                            inp = transform_normalize(inp).cuda()
                            out = self.net(inp)
                        else:
                            inp = orig_inp
                            if use_superpixel or self.pm.patch_size != 16:
                                if nums_to_exclude == 0:
                                    out = self.net(inp)
                                else:
                                    patch_masks = self.pm.slow_mask_orders(patch_orders=patch_orders,
                                                                           nums_to_exclude=nums_to_exclude,
                                                                           images=x, superpixel_extension=superpixel_extension)
                                    out = self.net(inp, patch_mask=patch_masks)
                            else:
                                patch_masks = self.pm.get_patch_masks(patch_orders, nums_to_exclude).cuda()
                                out = self.net(inp, patch_mask=patch_masks)                            
                        out = ch.softmax(out, 1)
                        pred = out.argmax(-1)
                        if nums_to_exclude == 0:
                            orig_label = pred
                        mini_accs.append((pred == orig_label).int().cpu())
                        orig_label_score = torch.gather(out, 1, orig_label.unsqueeze(1)).squeeze(1)
                        mini_soft_labels.append(orig_label_score.cpu())
                        mini_preds.append(pred.cpu())
                    accs.append(torch.stack(mini_accs))
                    soft_labels.append(torch.stack(mini_soft_labels))
                    preds.append(torch.stack(mini_preds))
                    debugs.append({'patch_orders': patch_orders})
                all_accs[m] = torch.cat(accs, dim=1)
                all_soft_labels[m] = torch.cat(soft_labels, dim=1)
                all_preds[m] = torch.cat(preds, dim=1)
                all_debugs[m] = debugs
        return all_accs, all_soft_labels, all_preds, all_debugs
    