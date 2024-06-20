import itertools

import torch
import numpy as np

from .loss import CLIPLoss
from .simdecoder import MIMHead
from .maedecoder import MAEHead
from .clip_extractor import ClipExtractor
from .medsam_extractor import MedSamExtractor
from .sam_extractor import SamExtractor


def get_params_group(model):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def maybe_add_full_model_gradient_clipping(optim, clip_norm_val=10):  # optim: the optimizer class
    # detectron2 doesn't have full model gradient clipping now

    enable = clip_norm_val > 0.0

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer if enable else optim

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask