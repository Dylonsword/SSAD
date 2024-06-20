from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import NestedTensor
from models.dino.blocks import FeatureFusionBlock_custom
from models.dino.vit import _make_pretrained_vits16_224, forward_vit


def _make_fusion_block(features, use_bn, expand, act=nn.ReLU(False), scale_factor=2):
    return FeatureFusionBlock_custom(
        features,
        act,
        deconv=False,
        bn=use_bn,
        expand=expand,
        align_corners=True,
        scale_factor=scale_factor
    )
    
    
def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch

class DenseVit(nn.Module):
    def __init__(self, pretrained, scratch, out_indices=[1,2,3], channels_last=False) -> None:
        super(DenseVit, self).__init__()
        self.pretrained, self.scratch = pretrained, scratch
        # for quick reshape and contiguous
        self.channels_last = channels_last
        # DINO need to construct the project layer
        self.expand = getattr(scratch, f"refinenet{1}").expand
        scale = 1
        if self.expand:
            scale = 2
        self.num_features = [getattr(scratch, f"refinenet{i}").features // scale for i in range(1, 5)]
        self.out_indices = out_indices

        # collect attention map
        self.attention = {}
        
    @classmethod
    def _dense_vit_init(self, **kwargs):
        if kwargs.pop("model_name") == "denseT_S_224":
            input_channels = kwargs.pop("input_channels")
            features = kwargs.pop("features")
            groups = kwargs.pop("groups")
            expand = kwargs.pop("expand")
            return_interm_indices = kwargs.pop("return_interm_indices")
            pretrained = _make_pretrained_vits16_224(**kwargs)
            
            scratch = _make_scratch(input_channels,
                                    features, groups=groups, expand=expand)
            if expand:
                features = features * 8
            for idx in range(4, 0, -1):
                scratch.add_module(f"refinenet{idx}", _make_fusion_block(features, use_bn=True, expand=expand, act=nn.ReLU(False)))
                if expand:
                    features = features // 2
                    
            return self(pretrained, scratch, return_interm_indices)

    def get_attention(self, name):
        attention = self.attention
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = (
                module.qkv(x)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * module.scale

            attn = attn.softmax(dim=-1)  # [:,:,1,1:]
            attention[name] = attn

        return hook

    def forward_attention(self, tensor_list: NestedTensor, hooks):
        """forward vit to get attention at each bolck idex
        """
        # self.attention = {}
        # register forward hook
        for i, hook in enumerate(hooks):
            self.pretrained.model.blocks[hook].attn.register_forward_hook(self.get_attention(f"attn_{i}"))
        
        x = tensor_list.tensors
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        # 128, 96, 32, 16
        # import pdb; pdb.set_trace()
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        # 128, 96, 32, 16
        # import pdb; pdb.set_trace()
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        outs = list()
        for idx, o in enumerate([path_1, path_2, path_3, path_4]):
            if idx in self.out_indices:
                outs.append(o)
        outs = tuple(outs)
        # out:
        # torch.Size([1, 128, 256, 256])
        # torch.Size([1, 256, 128, 128])
        # torch.Size([1, 512, 64, 64])
        # torch.Size([1, 1024, 32, 32])
        
        # collect for nesttensors        
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict

    
def build_dense_vit(model_name, 
                    img_size: int,
                    in_chans: int,  
                    enable_attention_hooks: bool=False, 
                    pretrain: str=None,
                    return_interm_indices=[1,2,3]):
    model_dict = {
        'denseT_S_224': dict(
            input_channels=[96, 192, 384, 768],
            features=256,
            hooks=[2, 5, 8, 11],
            use_readout="project",
            groups=1,   # scrach conv groups
            expand=True,    # hierachical channel numbers
        ),
        
    }
    
    base_dict = model_dict[model_name]
    # init input args
    base_dict["img_size"] = img_size
    base_dict["model_name"] = "denseT_S_224"
    base_dict["in_chans"] = in_chans
    base_dict["enable_attention_hooks"] = enable_attention_hooks
    base_dict["pretrain"] = pretrain
    base_dict["return_interm_indices"] = return_interm_indices
    print(base_dict)
    return DenseVit._dense_vit_init(**base_dict)


if __name__ == "__main__":
    dpt_backbone = build_dense_vit("denseT_S_224", in_chans=3, enable_attention_hooks=False, pretrain='')
    in_ten = torch.randn((1, 3, 512, 512))
    y = dpt_backbone(in_ten)
    import pdb;pdb.set_trace()
