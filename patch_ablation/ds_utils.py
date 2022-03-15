import sys
import os
sys.path.append('src')

from robustness.tools.label_maps import CLASS_DICT
import torch as ch
import torch
from robustness import model_utils, datasets
import torchvision
from matplotlib.colors import LinearSegmentedColormap
import gpustat
from config import cfg
from models import build_model
from skimage.segmentation import quickshift
from torchvision.datasets.folder import pil_loader
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import gc
import pickle as pkl
import torch.nn as nn

NOISE_SCALE = 20
DEFAULT_CMAP = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

TRANSFORM_NORMALIZE = torchvision.transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

CIFAR_NORMALIZE = torchvision.transforms.Normalize(
    mean=ch.tensor([0.4914, 0.4822, 0.4465]),
    std=ch.tensor([0.2023, 0.1994, 0.2010]),
)

TO_PIL = torchvision.transforms.ToPILImage()
DATA = 'ImageNet'
DATA_PATH_DICT = {
    'CIFAR': '/scratch/datasets/cifar10',
    'RestrictedImageNet': '/mnt/nfs/datasets/imagenet-pytorch',
    'ImageNet': '/mnt/nfs/datasets/imagenet-pytorch',
    'H2Z': '/scratch/datasets/A2B/horse2zebra',
    'A2O': '/scratch/datasets/A2B/apple2orange',
    'S2W': '/scratch/datasets/A2B/summer2winter_yosemite',
    'ImageNetAnnot': '/data/theory/robustopt/datasets/imagenet-bounding-boxes'
}
LABEL_MAP = CLASS_DICT[DATA]

NUM_WORKERS = 4
BSZ = 256 # 256

def get_loader(ds, bsz=BSZ):
    return torch.utils.data.DataLoader(ds, shuffle=False, batch_size=bsz, num_workers=NUM_WORKERS)

def print_gpu_stat():
    print(f'CUDA available: {ch.cuda.is_available()}')
    print(f'Device count: {ch.cuda.device_count()}')
    print(f'{ch.cuda.get_device_name(0)}')
    ch.cuda.get_device_properties('cuda')

    gpustat.print_gpustat()

    t = ch.cuda.get_device_properties(0).total_memory
    r = ch.cuda.memory_reserved(0) 
    a = ch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'Used by pytorch: {f/1e9:.2f} GB / {t/1e9:.2f} GB')
    
def print_alloc():
    ch.cuda.empty_cache()
    print_gpu_stat()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def get_bb_imagenet_ds(do_mask):
    in_file_path = os.path.join(DATA_PATH_DICT['ImageNet'], 'val')
    annot_folder = os.path.join(DATA_PATH_DICT['ImageNetAnnot'], 'val')
    IMG_TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])
    MASK_TRANSFORM = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
    ])
    
    def get_annotation(annot_file):
        tree = ET.parse(annot_file)
        root = tree.getroot()
        objects = root.findall('object')
        height = int(root.find('size').find('height').text)
        width = int(root.find('size').find('width').text)
        mask = torch.zeros(height, width)
        for i, obj in enumerate(objects):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            mask[ymin:(ymax+1), xmin:(xmax+1)] = i+1
        return mask.unsqueeze(0)
    
    def loader_fn(path):
        base_name = os.path.basename(path).split('.JPEG')[0] + ".xml"
        annot_file = os.path.join(annot_folder, base_name)
        img = IMG_TRANSFORM(pil_loader(path))
        if do_mask:
            mask = MASK_TRANSFORM(get_annotation(annot_file))
        else:
            mask = torch.zeros(1, img.shape[1], img.shape[2])
        return torch.cat([mask, img], dim=0)
    
    ds = torchvision.datasets.ImageFolder(in_file_path, loader=loader_fn)
    return ds
        
class ImageNetMegaDS(torch.utils.data.Dataset):
    def __init__(self, do_mask=False, factor=5, cache_file=None):
        self.dataset = get_bb_imagenet_ds(do_mask=do_mask)
        self.lime = None
        self.saliency = None
        self.superpixel = None
        self.factoring = np.arange(0, len(self.dataset), factor)
        print("length of DS", len(self.factoring))
        self.cache = None
        if cache_file is not None:
            self.cache = torch.load(cache_file)
            print("loaded cache")
        
    def __len__(self):
        return len(self.factoring)
    
    def save_file(self, filename):
        all_x = []
        all_y = []
        
        for idx in self.factoring:
            if idx % 50 == 0:
                print(idx)
            x, y = self.dataset[idx]
            all_x.append(x.cpu())
            all_y.append(y)
        all_x = torch.stack(all_x)
        all_y = torch.tensor(all_y)
        torch.save({'x': all_x,'y': all_y}, filename)
        
    def __getitem__(self, idx):
        if self.cache is None:
            orig_idx = self.factoring[idx]
            x, y = self.dataset[orig_idx]
        else:
            x, y = self.cache['x'][idx], self.cache['y'][idx]
        out = {
            'mask': x[0],
            'image': x[1:],
            'labels': y,
            'idx': idx,
        }
        if self.lime is not None:
            out['lime'] = self.lime[idx]
        if self.saliency is not None:
            for k in self.saliency.keys():
                out[f"saliency_{k}"] = self.saliency[k][idx]
        if self.superpixel is not None:
            out['superpixel'] = self.superpixel[idx]
        return out

class CIFARMegaDS(torch.utils.data.Dataset):
    def __init__(self, do_mask=False, factor=1, cache_file=None):
        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224))]
        )
        self.dataset = torchvision.datasets.CIFAR10('/mnt/nfs/datasets/cifar', train=False, transform=img_transform)
        self.factoring = np.arange(0, len(self.dataset), factor)
        print("length of DS", len(self.factoring))
        self.cache = None
        if cache_file is not None:
            self.cache = torch.load(cache_file)
            print("loaded cache")
        self.lime = None
        self.saliency = None
        self.superpixel = None
        
    def __len__(self):
        return len(self.factoring)
    
    def save_file(self, filename):
        all_x = []
        all_y = []
        
        for idx in self.factoring:
            if idx % 50 == 0:
                print(idx)
            x, y = self.dataset[idx]
            all_x.append(x.cpu())
            all_y.append(y)
        all_x = torch.stack(all_x)
        all_y = torch.tensor(all_y)
        torch.save({'x': all_x,'y': all_y}, filename)
        
    def __getitem__(self, idx):
        if self.cache is None:
            orig_idx = self.factoring[idx]
            x, y = self.dataset[orig_idx]
        else:
            x, y = self.cache['x'][idx], self.cache['y'][idx]
        out = {
            'image': x,
            'labels': y,
            'idx': idx,
        }
        if self.lime is not None:
            out['lime'] = self.lime[idx]
        if self.saliency is not None:
            for k in self.saliency.keys():
                out[f"saliency_{k}"] = self.saliency[k][idx]
        if self.superpixel is not None:
            out['superpixel'] = self.superpixel[idx]
        return out
    
    
def extend_superpixel(superpixels, total_patches):
    # all_superpixels is N x 224 x 224
    extension = torch.stack([superpixels == i for i in range(total_patches)], dim=1)
    return extension

ROBUSTNESS_MODELS = ['resnet18', 'resnet50', 'deit_small_patch16_224_retrain', 'robust_resnet50']
# Import the model deit
def get_model(cfg, model_path, orig_arch_name, prebuilt_model=None):
    print(model_path)
    if orig_arch_name not in ROBUSTNESS_MODELS:
        cfg.MODEL.MODEL_PATH = model_path
        cfg.MODEL.PRETRAINED = False
        net = build_model(cfg)
        checkpoint = torch.load(cfg.MODEL.MODEL_PATH , map_location="cpu")
        if 'model' in checkpoint:
            net.load_state_dict(checkpoint["model"])
        else:
            no_module_dict = {}
            if "net" in checkpoint:
                checkpoint = checkpoint["net"]
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                no_module_dict[k] = v
            net.load_state_dict(no_module_dict)
    else:
        dataset = getattr(datasets, DATA)(DATA_PATH_DICT[DATA])
        model_kwargs = {
            'arch': cfg.MODEL.ARCH if prebuilt_model is None else prebuilt_model,
            'add_custom_forward': orig_arch_name == 'deit_small_patch16_224_retrain',
            'dataset': dataset,
            'pytorch_pretrained': True,
            'resume_path': model_path
        }
        net, _ = model_utils.make_and_restore_model(**model_kwargs)
        net = net.model
        print('Loaded robustness lib model')
    net = net.eval().cuda()
    ch.cuda.empty_cache()
    print_gpu_stat()
    return net

def get_full_model(arch, model_path, special_config_path=None):
    cfg.defrost()
    model_dirs = {
        'resnet50_deitaug': '/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/resnet50_deitaug',
        'resnet18_deitaug': '/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/resnet18_deitaug',
        'deit_small_normfalse': '/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/deit_small_normfalse',
        'deit_small_resnet_aug': '/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/deitsmall_resnetaug',
        'deit_tiny_resnet_aug': '/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/deittiny_resnetaug',
    }
    model_dir = model_dirs.get(arch)
    if special_config_path is not None:
        CONFIG_FILE = special_config_path
    else:
        CONFIG_FILE = '../src/config/deit.yaml'
    assert os.path.exists(CONFIG_FILE)
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.ARCH = arch
    
    
    if model_dir is not None:
        with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
            custom_cfg = yaml.full_load(f)
        cfg.MODEL.ARCH = custom_cfg['MODEL']['ARCH']
        if 'TRANSFORMER' in custom_cfg['MODEL']:
            cfg.MODEL.TRANSFORMER.NORM_EMBED = custom_cfg['MODEL']['TRANSFORMER']['NORM_EMBED']
    if 'resnet18' == arch:
        REPRESENTATION_SIZE = 512
        if model_path is None:
            MODEL_PATH = '/data/theory/robustopt/hadi_shared_results/imagenet_experiments/opensource_models/resnet18_l2_eps0.ckpt'
        else:
            MODEL_PATH = model_path
        resnet = get_model(cfg, MODEL_PATH, arch)
        return resnet
    elif 'resnet50' == arch:
        REPRESENTATION_SIZE = 512
        if model_path is None:
            MODEL_PATH = '/data/theory/robustopt/hadi_shared_results/imagenet_experiments/opensource_models/resnet50_l2_eps0.ckpt'
        else:
            MODEL_PATH = model_path
        resnet = get_model(cfg, MODEL_PATH, arch)
        return resnet
    elif 'robust_resnet50' == arch:
        REPRESENTATION_SIZE = 512
        cfg.MODEL.ARCH = 'resnet50'
        if model_path is None:
            MODEL_PATH = '/data/theory/robustopt/hadi_shared_results/imagenet_experiments/opensource_models/resnet50_l2_eps3.ckpt'
        else:
            MODEL_PATH = model_path
        resnet = get_model(cfg, MODEL_PATH, arch)
        return resnet
    elif 'inceptionv3' == arch:
        return torchvision.models.inception_v3(pretrained=True).cuda().eval()
    elif 'vgg16' == arch:
        return torchvision.models.vgg16(pretrained=True).cuda().eval()
    elif 'deit_small_patch16_224_retrain' == arch:
        prebuilt_model = get_full_model('deit_small_patch16_224', None)
        cfg.MODEL.ARCH = 'deit_small_patch16_224'
        MODEL_PATH = model_path
        assert MODEL_PATH is not None
        transformer = get_model(cfg, MODEL_PATH, arch, prebuilt_model)
        return transformer.model
    elif 'deit_tiny_patch16_224' == arch:
        REPRESENTATION_SIZE = 192
        MODEL_PATH = '/data/theory/robustopt/saachij/src/CausalDomainTransfer/run/imagenet/old_deit_tiny_patch16_224-a1311bcf.pth'
        cfg.MODEL.ARCH = arch
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'deit_small_patch16_224' == arch:
        REPRESENTATION_SIZE = 192
        MODEL_PATH = '/data/theory/robustopt/saachij/src/CausalDomainTransfer/run/imagenet/old_deit_small_patch16_224-cd65a155.pth'
        cfg.MODEL.ARCH = arch
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'resnet50_deitaug' == arch:
        assert model_dir is not None
        MODEL_PATH = os.path.join(model_dir, 'model_best.pth')
        assert cfg.MODEL.ARCH == 'resnet50'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'resnet50_rerun' == arch:
        MODEL_PATH = '/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/rerun_random_seed/r50_model_state_dict.pt'
        cfg.MODEL.ARCH = 'resnet50'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'deit_small_rerun' == arch:
        REPRESENTATION_SIZE = 192
        MODEL_PATH = "/mnt/nfs/home/hadi/src/missingness/CausalDomainTransfer/run/imagenet/rerun_random_seed/deitsmall_model_state_dict.pt"
        cfg.MODEL.ARCH = 'deit_small_patch16_224'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'resnet18_deitaug' == arch:
        assert model_dir is not None
        REPRESENTATION_SIZE = 192
        MODEL_PATH = os.path.join(model_dir, 'model_best.pth')
        assert cfg.MODEL.ARCH == 'resnet18'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'deit_small_normfalse' == arch:
        assert model_dir is not None
        REPRESENTATION_SIZE = 192
        MODEL_PATH = os.path.join(model_dir, 'model_best.pth')
        assert cfg.MODEL.ARCH == 'deit_small_patch16_224'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'deit_small_resnet_aug' == arch:
        assert model_dir is not None
        REPRESENTATION_SIZE = 192
        MODEL_PATH = os.path.join(model_dir, 'model_best.pth')
        assert cfg.MODEL.ARCH == 'deit_small_patch16_224'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    elif 'deit_tiny_resnet_aug' == arch:
        assert model_dir is not None
        REPRESENTATION_SIZE = 192
        MODEL_PATH = os.path.join(model_dir, 'model_best.pth')
        assert cfg.MODEL.ARCH == 'deit_tiny_patch16_224'
        transformer = get_model(cfg, MODEL_PATH, arch)
        return transformer
    else:
        raise NotImplementedError()
        
def get_cifar_model(arch_name):
    assert arch_name in ['resnet50', 'deit']
    if arch_name == 'deit':
        prebuilt = get_full_model('deit_small_patch16_224', None)
        prebuilt.head = nn.Linear(prebuilt.head.in_features, 10)
        path = "/mnt/nfs/home/hadi/src/certified-vit/outdir_missingness/cifar/cifar10-deit_small_patch16_224-30epochs/checkpoint.pt.best"
        arch = prebuilt
    else:
        arch = 'resnet50'
        path="/mnt/nfs/home/hadi/src/certified-vit/outdir_missingness/cifar/cifar10-resnet50-30epochs/checkpoint.pt.best"
    dataset = getattr(datasets, DATA)(DATA_PATH_DICT[DATA])
    dataset.num_classes = 10
    model_kwargs = {
        'arch': arch,
        'add_custom_forward': True,
        'dataset': dataset,
        'pytorch_pretrained': False,
        'resume_path': path,
    }
    net, _ = model_utils.make_and_restore_model(**model_kwargs)
    net = net.model
    if arch_name == 'deit':
        net = net.model
    return net
        
    
    