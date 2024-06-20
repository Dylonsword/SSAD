import torch
import torch.nn as nn
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .fcos_vitB_vitdet import model
from models.auxiliaryHead import MAEHead, CLIPLoss, MedSamExtractor, get_params_group


# another params from file "base.py"
base = model_zoo.get_config("common/base.py")
# component
train = base.train
# redefine train params
train.output_dir = "train_out/disease/fcos/vitdet-ssat-medsam"
dataloader = base.dataloader
optimizer = base.optimizer
lr_multiplier = base.lr_multiplier
model = model

# SSAT
ssat = base.ssat
# define SSAT components
size2config = model_zoo.get_config("common/ssat_params.py").size2config
# SSAT models
num_patches = 1024  # (512 // 16)**2
ssat.model = L(MAEHead)(
    in_chans=3,
    patch_size=16,
    num_patches=num_patches,
    **size2config["B_4"],
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    pretrain=None,
    clip_criterion=None,
    output_rgb=True,    # 'False' means supervise in feature space
    use_cls_token=model.backbone.net.pretrain_use_cls_token
)
# SSAT optimizer
ssat.optimizer = L(torch.optim.SGD)(
        params=L(get_params_group)(
            model=ssat.model
        ),
        lr=1e-4,
        momentum=0.9,
        weight_decay=0.5,
)

# SSAT schedular
ssat.scheduler = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[1846, 5112],    # 26e, 72e
        num_updates=train.max_iter,
    ),
    warmup_length=200 / train.max_iter, # 200
    warmup_factor=0.001,
)

# TC loss
clip_params = dict(
    using_pixel_list=True,
    sam_size="vit_b",
    neck=False,
    sam_checkpoint="/data/xinyuan/tooth_disease_detection/3dSeg/MedSAM/sam_vit_b_01ec64.pth",
    struc_lambda=1, # drop
    distri_lambda=1,
)

clip_criterion = L(CLIPLoss)(
    args=clip_params,
    clip_extractor=L(MedSamExtractor)(
        args=clip_params
    ),
    using_pixel_list=True
)
ssat.model.clip_criterion = clip_criterion

# ssat hyp
ssat.enabled = True
ssat.mask_ratio = 0.6
ssat.method = "mae"
ssat.task_m = "vitdet"