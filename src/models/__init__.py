import logging
import torch.nn as nn
import torchvision.models as tvmodels


from .resnet import *
from .vision_transformer import *


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias)

    def forward(self, x):
        return self.fc1(x.view(x.size(0), -1))


def freeze_backbone(net, freeze_at):
    if freeze_at < 0:
        return
    logging.info("Freeze parameters at level {}".format(freeze_at))
    for stage_index in range(freeze_at):
        if stage_index == 0:
            m = net.conv1  # stage 0 is the conv1 + bn1
            for p in net.bn1.parameters():
                p.requires_grad = False
        else:
            m = getattr(net, "layer" + str(stage_index))
        for p in m.parameters():
            p.requires_grad = False


def build_model(cfg):
    model_names = sorted(name for name in tvmodels.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(tvmodels.__dict__[name]))
    print("torchvision models: \n", model_names)

    
    vitmodeldict = {
        "vit_small_patch16_224": vit_small_patch16_224,
        "vit_base_patch16_224": vit_base_patch16_224,
        "vit_base_patch16_384": vit_base_patch16_384,
        "vit_base_patch32_384": vit_base_patch32_384,
        "vit_large_patch16_224": vit_large_patch16_224,
        "vit_large_patch16_384": vit_large_patch16_384,
        "vit_large_patch32_384": vit_large_patch32_384,
        "vit_huge_patch16_224": vit_huge_patch16_224,
        "vit_huge_patch32_384": vit_huge_patch32_384,
        'deit_tiny_patch16_224': deit_tiny_patch16_224,
        'deit_small_patch16_224': deit_small_patch16_224,
        'deit_base_patch16_224': deit_base_patch16_224,
        'deit_base_patch16_384': deit_base_patch16_384,
    }
    vit_model_names = list(vitmodeldict.keys())
    print("Vision Transformer models: \n", vit_model_names)


    print('==> Building model..')
    if cfg.MODEL.ARCH in model_names:
        logging.info("Use torchvision predefined model")
        if cfg.MODEL.PRETRAINED:
            logging.info("=> using pre-trained model '{}'".format(cfg.MODEL.ARCH))
            net = tvmodels.__dict__[cfg.MODEL.ARCH](pretrained=True,)
            if net.fc.out_features != cfg.DATA.NUM_CLASSES:
                net.fc = nn.Linear(net.fc.in_features, cfg.DATA.NUM_CLASSES)
        else:
            logging.info("=> creating model '{}'".format(cfg.MODEL.ARCH))
            net = tvmodels.__dict__[cfg.MODEL.ARCH](num_classes=cfg.DATA.NUM_CLASSES)
    elif cfg.MODEL.ARCH in vit_model_names:
        logging.info("Use vision transformer model")
        if cfg.MODEL.PRETRAINED:
            logging.info("=> using pre-trained model '{}'".format(cfg.MODEL.ARCH))
            net = vitmodeldict[cfg.MODEL.ARCH](pretrained=True,
                                               drop_rate=cfg.MODEL.TRANSFORMER.DROP,
                                               drop_path_rate=cfg.MODEL.TRANSFORMER.DROP_PATH,
                                               norm_embed=cfg.MODEL.TRANSFORMER.NORM_EMBED)
            if net.num_classes != cfg.DATA.NUM_CLASSES:
                net.head = nn.Linear(net.embed_dim, cfg.DATA.NUM_CLASSES)
        else:
            logging.info("=> creating model '{}'".format(cfg.MODEL.ARCH))
            net = vitmodeldict[cfg.MODEL.ARCH](num_classes=cfg.DATA.NUM_CLASSES,
                                               drop_rate=cfg.MODEL.TRANSFORMER.DROP,
                                               drop_path_rate=cfg.MODEL.TRANSFORMER.DROP_PATH,
                                               norm_embed=cfg.MODEL.TRANSFORMER.NORM_EMBED)
    else:
        raise ValueError(
            "Unimplemented model architecture: {}".format(cfg.MODEL.ARCH))

    if 'resnet' in cfg.MODEL.ARCH:
        freeze_backbone(net, cfg.MODEL.FREEZE_CONV_BODY_AT)

    return net

