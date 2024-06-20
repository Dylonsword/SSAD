import torch.nn as nn
from functools import partial

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7

from .fcos_fpn import model


# another params from file "base.py"
base = model_zoo.get_config("common/base.py")
# component
train = base.train
dataloader = base.dataloader
optimizer = base.optimizer
lr_multiplier = base.lr_multiplier
# ssat
ssat = base.ssat


del model.backbone
# redefine a backbone
# Base ViT
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=512,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        pretrain_use_cls_token=True
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    # top_block=L(LastLevelMaxPool)(),
    top_block=L(LastLevelP6P7)(in_channels="${..out_channels}", out_channels="${..out_channels}", in_feature="p5"),
    norm="LN",
    square_pad=512,
)

# in feat
model.head_in_features = ["p3", "p4", "p5", "p6", "p7"]
# cls
model.num_classes=4