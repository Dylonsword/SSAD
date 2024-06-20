# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from util.misc import NestedTensor, clean_state_dict, is_main_process

from .convnext import build_convnext
from .dense_vit import build_dense_vit, DenseVit
from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer, SwinTransformer
from .vitdet import SimpleFeaturePyramid, build_vitdet_backbone

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
        in_chans: int = 3,
        pretrain: str = "",
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation], norm_layer=batch_norm, pretrained=False
            )
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)
        self.in_chans = in_chans
        # layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.num_feature = [256, 512, 1024, 2048]

        # load pretrain
        if pretrain:
            self.load_pretrain(pretrain, backbone)

    def load_pretrain(self, pretrain, backbone):
        checkpoint = torch.load(pretrain, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        cur_model_checkpoint = backbone.state_dict()
        drop_key = []
        for k, v in checkpoint.items():
            if k.startswith("encoder"):
                k = k[8:]
                
                if k in cur_model_checkpoint:
                    if cur_model_checkpoint[k].size() == v.size():
                        cur_model_checkpoint[k] = v
                    else:
                        drop_key.append(k)
                else:
                        drop_key.append(k)
            else:
                drop_key.append(k)
        
        print(f"all drop keys: {drop_key}")
        load_msg = backbone.load_state_dict(cur_model_checkpoint, strict=False)
        print(f"load msg: {load_msg}")
        print("backbone ResNet pretrain loadding is done")
        return backbone


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        
        if isinstance(backbone, SwinTransformer):
            # hyparams
            self.num_feature = backbone.num_features
            self.patch_embed = backbone.patch_embed
            self.ape = backbone.ape
            self.patch_size = backbone.patch_size
            self.in_chans = backbone.in_chans
            # module
            if self.ape:
                self.absolute_pos_embed = backbone.absolute_pos_embed
            self.out_indices = backbone.out_indices
            self.pos_drop = backbone.pos_drop
            self.layers = backbone.layers
            for indx in backbone.out_indices:
                self.add_module(f"norm{indx}", getattr(backbone, f'norm{indx}'))
            self.forward_attention = backbone.forward_attention # function

        elif isinstance(backbone, DenseVit):
            # import pdb; pdb.set_trace()
            # hyp param
            self.in_chans = backbone.pretrained.model.in_chans
            self.num_feature = [backbone.pretrained.model.embed_dim]
            self.start_index = backbone.pretrained.model.start_index
            self.num_patches = backbone.pretrained.model.patch_embed.num_patches
            self.patch_size = backbone.pretrained.model.patch_size
            self.attention = backbone.attention
            # layers
            self.pretrained = backbone.pretrained.model
            self.patch_embed = backbone.pretrained.model.patch_embed
            self.pos_embed = backbone.pretrained.model.pos_embed
            self.dist_token = getattr(backbone.pretrained.model, "dist_token", None)
            self.cls_token = backbone.pretrained.model.cls_token
            self.pos_drop = backbone.pretrained.model.pos_drop
            self.blocks = backbone.pretrained.model.blocks
            self.norm = backbone.pretrained.model.norm
            # function
            self.forward_attention = backbone.forward_attention
        
        elif isinstance(backbone, SimpleFeaturePyramid):
            self.backbone=backbone
            self.patch_size = 16
            self.num_patches = self.backbone.net.num_patches

        elif isinstance(backbone, Backbone):
            self.patch_size = 4
            self.in_chans = backbone.in_chans
            self.num_feature = backbone.num_feature
            # layers
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)
    pretrained_checkpoint = getattr(args, "pretrained_checkpoint", "")
    
    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone, train_backbone, args.dilation, return_interm_indices, batch_norm=FrozenBatchNorm2d, in_chans=args.in_chans,
            pretrain=args.pretrained_checkpoint
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in ["swin_T_224_1k", "swin_B_224_22k", "swin_B_384_22k", "swin_L_224_22k", "swin_L_384_22k"]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=args.dilation,
            use_checkpoint=use_checkpoint,
            pretrained_checkpoint=pretrained_checkpoint
        )

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break
        if "backbone_dir" in args:
            pretrained_dir = args.backbone_dir
            PTDICT = {
                "swin_T_224_1k": "swin_tiny_patch4_window7_224.pth",
                "swin_B_384_22k": "swin_base_patch4_window12_384.pth",
                "swin_L_384_22k": "swin_large_patch4_window12_384_22k.pth",
            }
            pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
            checkpoint = torch.load(pretrainedpath, map_location="cpu")["model"]
            from collections import OrderedDict

            def key_select_function(keyname):
                if "head" in keyname:
                    return False
                if args.dilation and "layers.3" in keyname:
                    return False
                return True

            _tmp_st = OrderedDict({k: v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    elif args.backbone in ["convnext_xlarge_22k"]:
        backbone = build_convnext(
            modelname=args.backbone,
            pretrained=True,
            out_indices=tuple(return_interm_indices),
            backbone_dir=args.backbone_dir,
        )
        bb_num_channels = backbone.dims[4 - len(return_interm_indices) :]
    
    elif args.backbone in ["denseT_S_224", ]:
        backbone = build_dense_vit(
            img_size=args.img_size,
            in_chans=args.in_chans,
            model_name=args.backbone,
            enable_attention_hooks=False,
            pretrain=args.backbone_pretrain,
        )
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    
    elif args.backbone in ["vitdet_B"]:
        backbone = build_vitdet_backbone(args)
        
        bb_num_channels = [256] * len(return_interm_indices)

    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(
        type(bb_num_channels)
    )
    return model
