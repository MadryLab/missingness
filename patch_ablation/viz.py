import numpy as np
import patch_ablation.ds_utils as ds_utils
import patch_ablation.saliency_utils as sal_utils
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

def get_masked_batch(pm, x, init_sal_method='random', nums_to_exclude_list=None):
    # x should be unnormalized
    if init_sal_method == 'random':
        patch_orders = pm.generate_random_patch_orders(x.shape[0])
    else:
        inp_x = ds_utils.TRANSFORM_NORMALIZE(x).cuda()
        out = net(inp_x)
        out = ch.softmax(out, 1)
        score, pred = out.max(-1)
        attr = sal_utils.perform_saliency_batch_size(patch_master=pm, m=init_sal_method, inp=inp_x, pred=pred, orig_inp=x)
        _, patch_orders = get_patch_ordering(attr)
        if len(x) == 1:
            fig, ax = plt.subplots(1, 3, figsize=(4*3,4))
            saliency_map = np.transpose(attr[0].cpu().detach().numpy(), (1,2,0))
            original_img = np.transpose(x[0].cpu().detach().numpy(), (1,2,0))
            ax[0].imshow(original_img, interpolation='nearest')  
            ax[0].axis('off')
            viz.visualize_image_attr(saliency_map, original_img, "heat_map", "positive",
                                                    plt_fig_axis=(fig,ax[1]), cmap=default_cmap,
                                                    use_pyplot=False, show_colorbar=True,)
            viz.visualize_image_attr(saliency_map, original_img, "blended_heat_map",
                                    "positive", plt_fig_axis=(fig,ax[2]), cmap=default_cmap,
                                     use_pyplot=False, show_colorbar=True,)
            plt.show()
        
    if nums_to_exclude_list is None:
        nums_to_exclude_list = np.arange(197)
    all_x = []
    for nums_to_exclude in nums_to_exclude_list:
        if nums_to_exclude == 0:
            inp = x
        else:
            inp = mask_images(patch_orders, nums_to_exclude, x)
        all_x.append(inp)
    return torch.stack(all_x)

def disp_label(label):
    label = label.split(",")[0]
    label = '\n'.join(label.split(" "))
    return label

def do_vis_masking(X_, Y_, init_sal_method='random'):
    Y = Y_

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)
    num_images = 16
    methods = ['Saliency']
    full_saliencies = []
    for i in range(num_images):
        x_ = X_[[i]]
        y_ = Y_[[i]]
        imgs = get_masked_batch(x_, init_sal_method=init_sal_method,
                                nums_to_exclude_list=np.arange(0, 196, 32)).squeeze(1)            
        inp = transform_normalize(imgs).cuda()
        out = net(inp)
        out = ch.softmax(out, 1)
        score, pred = out.max(-1)
        all_saliencies = {}
        for m in methods:
            print(m)
            fig, ax = plt.subplots(3,len(imgs), figsize=(4*len(imgs),4*3))
            attr = perform_saliency_batch_size(m=m, inp=inp, pred=pred, orig_inp=imgs)
            all_saliencies_m = []
            for j in range(imgs.shape[0]):
                smap = attr[j].cpu().detach().numpy()
                saliency_map = np.transpose(smap, (1,2,0))
                original_img = np.transpose(imgs[j].cpu().detach().numpy(), (1,2,0))
                ax[0,j].imshow(original_img, interpolation='nearest')  
                ax[0,j].axis('off')
                gt_short = disp_label(label_map[y_.item()])
                pred_short = disp_label(label_map[pred[j].item()])
                title = f'GT = {gt_short}\n Pred = {pred_short}'
                ax[0,j].set_title(title)
                all_saliencies_m.append(smap)
                viz.visualize_image_attr(saliency_map, original_img, "heat_map",
                                                        "positive",
                                                        plt_fig_axis=(fig,ax[1,j]),
                                                        cmap=default_cmap,
                                                        use_pyplot=False,                                     
                                                        show_colorbar=True,)
                viz.visualize_image_attr(saliency_map, original_img, "blended_heat_map",
                                        "positive",
                                        plt_fig_axis=(fig,ax[2,j]),
                                        cmap=default_cmap,
                                        use_pyplot=False,                                     
                                        show_colorbar=True,)
            ch.cuda.empty_cache() 
            fig.tight_layout()
            plt.show() 
            all_saliencies[m] = all_saliencies_m
        full_saliencies.append(all_saliencies)
    return full_saliencies

def do_vis():
    data_iterator = enumerate(test_loader)
    _, (X,Y) = next(data_iterator)
    num_images = 8

    methods = ['IntegratedGradients','Saliency','GuidedBackprop', 'InputXGradient']
    methods = ['Saliency']
    # methods = ['DeepliftShap', 'Lime']
    all_saliency_maps = {}
    for m in methods:
        print(f'Method: {m}')
        fig, ax = plt.subplots(2,num_images, figsize=(20,10))
        i = 0
        sal_maps = []
        for x_, y_ in zip(X,Y):
            orig_inp = x_.unsqueeze(0)
            inp = transform_normalize(orig_inp).cuda()
            y_ = y_.cuda()
            out = net(inp)
            out = ch.softmax(out, 1)
            score, pred = out.max(-1)
            # output = F.softmax(output, dim=1)
            # prediction_score, pred_label_idx = torch.topk(output, 1)

            print(f'Pred: {label_map[pred.item()]} | Label: {label_map[y_.item()]} | Score: {score.item()}')
            attr = perform_saliency(m=m, inp=inp, pred=pred, orig_inp=orig_inp)
            sal_maps.append(attr.squeeze().cpu().detach().numpy())
            saliency_map = np.transpose(attr.squeeze().cpu().detach().numpy(), (1,2,0))
            original_img = np.transpose(x_.squeeze().cpu().detach().numpy(), (1,2,0))
            ax[0,i].imshow(original_img, interpolation='nearest')    
            ax[0,i].axis('off')
            ax[0,i].set_title(f'GT = {label_map[y_.item()].split(",")[0]}\n'
                        f'Pred = {label_map[pred.item()].split(",")[0]} ({score.item():.2f})')
            viz.visualize_image_attr(saliency_map, original_img, "heat_map",
                                                    "positive",
                                                    plt_fig_axis=(fig,ax[1,i]),
                                                    cmap=default_cmap,
                                                    use_pyplot=False,                                     
                                                    show_colorbar=True,)
            ch.cuda.empty_cache()
            i += 1
            if i == num_images:
                break
        sal_maps = np.stack(sal_maps)
        all_saliency_maps[m] = sal_maps
        plt.subplots_adjust(wspace=0.04, hspace=-.5)
        plt.show()
