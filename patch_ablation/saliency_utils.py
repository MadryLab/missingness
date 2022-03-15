from models.vision_transformer import Debug_Flags
import ds_utils
import torch as ch
import torch
from captum.attr import IntegratedGradients, GradientShap, Lime, Saliency, GuidedBackprop, DeepLift
from captum.attr import Occlusion, InputXGradient, DeepLiftShap
from captum.attr import NoiseTunnel
from patch_ablation_utils import PatchMaster

BATCH_SIZES = {
    'IntegratedGradients': 2,
    'Saliency': 32,
    'GuidedBackprop': 8,
    'InputXGradient': 8,
    'Influence': 500,
    'InfluenceMask': 250,
    'InfluenceScore': 500,
    'InfluenceMaskScore': 250,
}

class PatchTransformerInfluenceCalculator:
    def __init__(self, patch_master, norm_scheme='none', T=5000):
        self.norm_scheme = norm_scheme
        self.T = T
        self.patch_master = patch_master
        
    def perform_masked_forward_pass(self, patches_to_include, expanded_patch_masks, net, orig_inp, normalized_inp, orig_pred):
        out, debug = net(normalized_inp, debug_flags=[Debug_Flags.FINAL_LATENT_VECTOR], patch_mask=expanded_patch_masks)
        return out, debug['FINAL_LATENT_VECTOR']
    
    def get_score(self, one_mask, zero_mask, b_final_latents):
        return (b_final_latents[one_mask].mean(dim=0) - b_final_latents[zero_mask].mean(dim=0)).norm()
    
    def normalize(self, all_l2_scores):
        if self.norm_scheme == 'softmax': # softmax
            all_l2_scores = ch.softmax(all_l2_scores, -1)
        elif self.norm_scheme == 'minmax': # min max scale
            all_l2_scores -= all_l2_scores.min(1, keepdim=True)[0]
            all_l2_scores /= all_l2_scores.max(1, keepdim=True)[0]
        elif self.norm_scheme == 'none':
            pass
        return all_l2_scores
    
    def get_patch_influences(self, orig_inp, normalized_inp, net): # T is number of trials, k is patches per trial
        T = self.T
        with torch.no_grad():
            orig_out = net(normalized_inp)
            orig_logits = ch.softmax(orig_out, 1)
            _, orig_pred = orig_logits.max(-1)
            
            B = normalized_inp.shape[0]
            all_patch_masks, final_latents = [], []
            for i in range(T):
                if i % 200 == 0:
                    print(i)
                patches_to_include = np.arange(196)[np.random.randint(0, 2, size=196) > 0]
                patch_mask = np.concatenate([[0], patches_to_include + 1], axis=0)
                all_patch_masks.append(patch_mask)
                patch_mask = torch.tensor(patch_mask).long()
                patch_mask = patch_mask.expand(B, -1).cuda()
                out, latent = self.perform_masked_forward_pass(patches_to_include, 
                                                              patch_mask, 
                                                              net,
                                                              orig_inp, 
                                                              normalized_inp,
                                                              orig_pred)
                final_latents.append(latent.cpu())
            final_latents = torch.stack(final_latents, 0)
            full_mask = torch.zeros(T, 197)
            for i in range(T):
                full_mask[i, all_patch_masks[i]] = 1
            all_l2_scores = []
            for b in range(B):
                b_final_latents = final_latents[:, b]
                l2_scores = []
                for idx in range(1, 197):
                    one_mask = full_mask[:, idx] == 1
                    zero_mask = full_mask[:, idx] == 0
                    l2_diff = self.get_score(one_mask, zero_mask, b_final_latents)
                    l2_scores.append(l2_diff.item())
                all_l2_scores.append(np.array(l2_scores))
            all_l2_scores = torch.tensor(np.stack(all_l2_scores))
            all_l2_scores = self.normalize(all_l2_scores)
            all_l2_scores = all_l2_scores.reshape(B, 14, 14)
            all_l2_scores = torch.repeat_interleave(torch.repeat_interleave(all_l2_scores, 16, 2), 16, 1)
            all_l2_scores = all_l2_scores.unsqueeze(1).expand(-1, 3, -1, -1)
        return all_l2_scores
    
class PatchTransformerMaskCalculator(PatchTransformerInfluenceCalculator):
    def perform_masked_forward_pass(self, patches_to_include, expanded_patch_masks, net, orig_inp, normalized_inp, orig_pred):
        patches_to_exclude = np.array([u for u in np.arange(self.pm.total_patches) if u not in patches_to_include])
        masked_inp = ds_utils.TRANFORM_NORMALIZE(batch_mask_image(orig_inp, patches_to_exclude))
        masked_inp = masked_inp.cuda()
        out, debug = net(masked_inp, debug_flags=[Debug_Flags.FINAL_LATENT_VECTOR])
        return out, debug['FINAL_LATENT_VECTOR']
    
class PatchResNetMaskCalculator(PatchTransformerInfluenceCalculator):
    def perform_masked_forward_pass(self, patches_to_include, expanded_patch_masks, net, orig_inp, normalized_inp, orig_pred):
        patches_to_exclude = np.array([u for u in np.arange(self.pm.total_patches) if u not in patches_to_include])
        masked_inp = ds_utils.TRANFORM_NORMALIZE(batch_mask_image(orig_inp, patches_to_exclude))
        masked_inp = masked_inp.cuda()
        out, latent = net(masked_inp, with_latent=True)
        return out, latent
    
class PatchTransformerInfluenceCalculatorScore(PatchTransformerInfluenceCalculator):
    def perform_masked_forward_pass(self, patches_to_include, expanded_patch_masks, net, orig_inp, normalized_inp, orig_pred):
        out = net(normalized_inp,patch_mask=expanded_patch_masks)
        logits = ch.softmax(out, 1)
        return out, logits[:, orig_pred]

class PatchMaskCalculatorScore(PatchTransformerInfluenceCalculatorScore):
    def perform_masked_forward_pass(self, patches_to_include, expanded_patch_masks, net, orig_inp, normalized_inp, orig_pred):
        patches_to_exclude = np.array([u for u in np.arange(self.pm.total_patches) if u not in patches_to_include])
        masked_inp = ds_utils.TRANFORM_NORMALIZE(batch_mask_image(orig_inp, patches_to_exclude))
        masked_inp = masked_inp.cuda()
        out = net(masked_inp)
        logits = ch.softmax(out, 1)
        return out, logits[:, orig_pred]
    
# 
# 
# 
# 

def perform_saliency(patch_master, net, m, inp, pred, orig_inp):
    if m == 'IntegratedGradients':
        integrated_gradients = IntegratedGradients(net)
        attributions_ig = integrated_gradients.attribute(inp, target=pred, n_steps=200)
        noise_tunnel = NoiseTunnel(integrated_gradients)
        attr = noise_tunnel.attribute(inp, nt_samples=1, nt_type='smoothgrad_sq', target=pred)
    elif m == 'Saliency':
        saliency = Saliency(net)
        attr = saliency.attribute(inp.float(), target=pred)
    elif m == 'GuidedBackprop':
        saliency = GuidedBackprop(net)
        attr = saliency.attribute(inp, target=pred)
    elif m == 'InputXGradient':
        saliency = InputXGradient(net)
        attr = saliency.attribute(inp, target=pred)
    elif m == 'DeepLift':
        saliency = DeepLift(net)
        attr = saliency.attribute(inp, target=pred)
    elif m == 'DeepLiftShap':
        saliency = DeepLiftShap(net)
        attr = saliency.attribute(inp, target=pred)
    elif m == 'Occlusion':
        saliency = Occlusion(net)
        attr = saliency.attribute(inp, target=pred, sliding_window_shapes=(3,3,3))
    elif m == 'Lime':
        saliency = Lime(net)
        attr = saliency.attribute(inp, target=pred, n_samples=200)
    elif 'Influence' in m:
        calculator_dict = {
            'Influence': PatchTransformerInfluenceCalculator,
            'InfluenceMask': PatchTransformerMaskCalculator if 'deit' in patch_master.arch else PatchResNetMaskCalculator,
            'InfluenceScore': PatchTransformerInfluenceCalculatorScore,
            'InfluenceMaskScore': PatchMaskCalculatorScore,
        }
        pi_class = calculator_dict[m](patch_master=patch_master, norm_scheme='minimax', T=1000)
        attr = pi_class.get_patch_influences(orig_inp=orig_inp, normalized_inp=inp, net=net)
    attr_return = attr.cpu().detach().clone()
    # cleanup
    del attr
    if 'Influence' not in m:
        if m == 'IntegratedGradients':
            del attributions_ig
            del integrated_gradients
        else:
            del saliency
    ch.cuda.empty_cache()
    return attr_return

def perform_saliency_batch_size(patch_master, net, m, inp, pred, orig_inp):
    indices = torch.arange(inp.shape[0])
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(indices), batch_size=BATCH_SIZES[m], shuffle=False)
    all_attrs = []
    for idx_vec in dl:
        all_attrs.append(perform_saliency(patch_master, net, m, inp[idx_vec], pred[idx_vec], orig_inp[idx_vec]))
    return torch.cat(all_attrs, dim=0)


