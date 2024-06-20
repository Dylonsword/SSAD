from functools import partial
from omegaconf import OmegaConf
from fvcore.common.param_scheduler import MultiStepParamScheduler

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate


train = OmegaConf.create()
# some Train params
train.output_dir=""
train.init_checkpoint="checkpoints/vitdet.pkl"
train.max_iter=7100     # 100 ep = 7100 iters * 8 images/iter / 564 images/ep
train.amp=dict(enabled=False)  # options for Automatic Mixed Precision
train.ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    )
train.checkpointer=dict(eval_period=71, val_metric="bbox/AP")    # options for BestCheckpointer
train.eval_period=71    # for Metric eval
train.log_period=71
train.device="cuda"
train.seed = 3467


# Dataloader
# Data using dentex
image_size = 512
dataloader = model_zoo.get_config("common/coco.py").dataloader
# dataloader.train.mapper.tfm_gens = [
#     L(T.RandomFlip)(horizontal=True),  # flip first
#     L(T.ResizeScale)(
#         min_scale=1.9, max_scale=2.5, target_height=image_size, target_width=image_size
#     ),
#     L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
# ]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 8

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = False
# dataloader.test.mapper.tfm_gens = [
#     L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
# ]

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# Schedular
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[1846, 5112],    # 26e, 72e
        num_updates=train.max_iter,
    ),
    warmup_length=100 / train.max_iter, # 100
    warmup_factor=0.001,
)

# If with SSAT
ssat = dict(
    enabled=False,
    model=None,    # Init by LazyCall
    optimizer=None,    # Init by LazyCall
    scheduler=None,     # Init by LazyCall
    loss_scale=0.1     # Init by LazyCall
)
