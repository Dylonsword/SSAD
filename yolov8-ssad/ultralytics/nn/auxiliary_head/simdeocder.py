import math
import copy
from typing import List, Dict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from auxiliary_head.basic_layer import ConvDecoder


# yolo
class MIMHead(nn.Module):
    def __init__(self, num_features, restruc_indices, encoder_stride, embed_dim, heavily=True, pretrained_checkpoint="", clip_criterion=None):
        super().__init__()
        encoder_stride = encoder_stride
        num_features = copy.deepcopy(num_features)
        self.num_features = num_features
        self.encoder_stride = encoder_stride
        self.restruc_indices =restruc_indices
        self.embed_dim = embed_dim
        self.heavily = heavily

        if heavily:
            self.decoder = self.build_heavy_decoder(num_features, wnrom=False)
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(in_channels=num_features[-1], out_channels=self.encoder_stride[-1] ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride[-1]),
            )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.mask_token, mean=0., std=.02)
        self.clip_criterion = clip_criterion

        if pretrained_checkpoint:
            if isinstance(pretrained_checkpoint, str):
                print("auxi head use pretrained checkpoint")
                self.load_pretrain(pretrained_checkpoint)
    
    def build_heavy_decoder(self, num_feat, wnrom=False):
        num_feat = copy.deepcopy(num_feat)
        num_feat.reverse()
        num_feat.append(num_feat[-1] // 2)
        return ConvDecoder(num_feat, wnrom)

    def diffdet_encode(self, x, mask, encoder):
        DiffdetTransformer = encoder

        x_tensor = DiffdetTransformer.patch_embed(x)

        assert mask is not None
        B, _, Wh, Ww = x_tensor.size()

        mask_tokens = self.mask_token.expand(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
        w = mask.unsqueeze(1).type_as(mask_tokens)
        x_tensor = x_tensor * (1. - w) + mask_tokens * w

        if DiffdetTransformer.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x_tensor = (x_tensor + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x_tensor = x_tensor.flatten(2).transpose(1, 2)
        x_tensor = DiffdetTransformer.pos_drop(x_tensor)

        outs = {}
        for i in range(DiffdetTransformer.num_layers):
            layer = DiffdetTransformer.layers[i]
            x_out, H, W, x_tensor, Wh, Ww = layer(x_tensor, Wh, Ww)

            # if i in self.restruc_indices:
            norm_layer = getattr(DiffdetTransformer, f'norm{i}', None)
            if norm_layer is not None:
                x_out = norm_layer(x_out)

            out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            # outs.append(out)
            outs['swin{}'.format(i)] = out

        return x, outs
 
    def resnet_encode(self, x, mask, encoder):
        if not isinstance(x, torch.Tensor):
            try:
                x = x.tensors
            except Exception:
                x = x.tensor
        ori_x = x
        # detectron resnet
        if getattr(encoder, "backbone", None) is not None:
            encoder = encoder.backbone.bottom_up
        stem = getattr(encoder, "stem", None)
        if stem is None:
            stem = encoder
        # downsample 4
        x = stem.conv1(x)
        bn1 = getattr(encoder, "bn1", None)
        if bn1 is not None:
            x = bn1(x)
        relu = getattr(encoder, "relu", None)
        if relu is not None:
            x = relu(x)
        else:
            x = F.relu(x)
        maxpool = getattr(encoder, "maxpool", None)
        if maxpool is not None:
            x = maxpool(x)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # import pdb;pdb.set_trace()
        B, _, H, W = x.size()
        # mask feature
        mask_tokens = self.mask_token.expand(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        w = mask.unsqueeze(1).type_as(mask_tokens)
        x_masked = x * (1. - w) + mask_tokens * w

        outs = {}
        # layer1 ~ layer4
        for i in range(1, 5):
            try:
                stage = getattr(encoder, f"stages")
                layer = encoder.stages[i-1]
            except:
                layer = getattr(encoder, f"layer{i}")
            x_masked = layer(x_masked)
            outs[f"layer{i-1}"] = x_masked
        return ori_x, outs

    def get_abs_pos(self, abs_pos, has_cls_token, hw):
        """
        Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
            dimension for the original embeddings.
        Args:
            abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
            has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
            hw (Tuple): size of input image tokens.

        Returns:
            Absolute positional embeddings after processing with shape (1, H, W, C)
        """
        h, w = hw
        if has_cls_token:
            abs_pos = abs_pos[:, 1:]
        xy_num = abs_pos.shape[1]
        size = int(math.sqrt(xy_num))
        assert size * size == xy_num

        if size != h or size != w:
            new_abs_pos = F.interpolate(
                abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            return abs_pos.reshape(1, h, w, -1)
        
    def vitdet_encode(self, x, mask, encoder):
        if not isinstance(x, torch.Tensor):
            try:
                x = x.tensors
            except Exception:
                x = x.tensor

        ori_x = x
        encoder = encoder.backbone.net
        x = encoder.patch_embed(x)
        B, Wh, Ww, C = x.shape

        mask_tokens = self.mask_token.expand(B, Wh, Ww, -1)
        w = mask.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if encoder.pos_embed is not None:
            x = x + self.get_abs_pos(
                encoder.pos_embed, encoder.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for blk in encoder.blocks:
            x = blk(x)

        return ori_x, x.permute(0, 3, 1, 2)    # b c h w

    def yolo_encode(self, x, mask, encoder):
        # only for yolo-P5
        # stem layer 0, then backbone end at 10
        # down sample 32 factor, take P3 P4 P5 
        # print("正在运行yolo encode")
        # stem layer
        x_tensor = x.clone()
        x_tensor = encoder[0](x_tensor) # B, 64, H/2, W/2  
        # import pdb;pdb.set_trace()
        assert len(mask.shape) == 3, f"mask shape is invalid, it may be initby random: {mask.shape}"

        # import pdb;pdb.set_trace()
        B, _, Wh, Ww = x_tensor.shape
        mask_tokens = self.mask_token.expand(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
        w = mask.unsqueeze(1).type_as(mask_tokens)
        x_masked = x_tensor * (1. - w) + mask_tokens * w

        outs = {}
        index = 1
        for i, m in enumerate(encoder[1:9]):
            x_masked = m(x_masked)
            if i % 2 != 0:   # stage1 to stage 4
                outs[f"layer{index}"] = x_masked
                index += 1
        # k: torch.Size([4, 128, 160, 160])
        # k: torch.Size([4, 256, 80, 80])
        # k: torch.Size([4, 512, 40, 40])
        # k: torch.Size([4, 512, 20, 20])
        return x, outs
    
    def get_encode_feat(self, x, mask, encoder, task_m="dino"):
        # if task_m == "dino":
        #     return self.dino_encode(x, mask, encoder)
        # elif task_m == "diffdet":
        #     return self.diffdet_encode(x, mask, encoder)
        # elif task_m == "densevit":
        #     return self.densevit_encode(x, mask, encoder)
        # elif task_m == "resnet":
        #     return self.resnet_encode(x, mask, encoder)
        # elif task_m == "vitdet":
        #     return self.vitdet_encode(x, mask, encoder)
        return self.yolo_encode(x, mask, encoder)
    
    def load_pretrain(self, pretrained_checkpoint):
        if isinstance(pretrained_checkpoint, str):
            drop_keys = []
            model_state = torch.load(pretrained_checkpoint, map_location="cpu")
            self_state = self.state_dict()
            for k, v in model_state.items():
                if k.startswith("module."):
                    k = k[7:]
                if k.startswith("decoder.") or "mask_token" in k:
                    if k.startswith("encoder."):
                        k = k[8:]
                    if k in self_state:
                        self_state[k] = v
                    else:
                        drop_keys.append(k)

            message = self.load_state_dict(self_state, strict=False)
            print(message)
            print(f"dorped key {drop_keys}")
            del model_state, self_state
            torch.cuda.empty_cache()
    
    def decoder_forward(self, x):
        x_recs = []
        # import pdb;pdb.set_trace()
        if self.heavily:
            if isinstance(x, (List, tuple)):
                x = x[-1]
            elif isinstance(x, Dict):
                last_idx = len(x) - 1
                last_x = x.get(f"swin{last_idx}", None)
                if last_x is not None:
                    x = last_x
                else:
                    x = x.get(f"layer{last_idx}", None)

            x_recs = [self.decoder(x)]
        else:
            x_recs = [self.decoder(x)]
        return x_recs
        
    def forward(self, imgs, masks, encoder, task_m, mask_ratio=0.6):
        if not encoder.training:
            return torch.zeros(1, device=imgs.device)
        x, z = self.get_encode_feat(imgs, masks, encoder, task_m)
        x_recs = self.decoder_forward(z)
        patch_size = encoder.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]

        in_chans = encoder.in_chans

        mask = masks.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1).contiguous().type_as(x)
        if not self.training:   # not training output, the reconstructed image
            return x_recs, mask
        
        loss = 0
        for i, x_rec in enumerate(x_recs):
            weight = 0.2 ** (i+1)  if not self.heavily else 1
            loss_recon = F.l1_loss(x, x_rec, reduction='none')
            loss += weight * (loss_recon * mask).sum() / (mask.sum() + 1e-5) / in_chans
            if self.clip_criterion is not None:
                if self.clip_criterion.using_pixel_list:
                    loss += (i+1) * self.clip_criterion(x * mask, x_rec * mask, [encoder.pixel_mean, encoder.pixel_std])
                else:
                    loss += (i+1) * self.clip_criterion(x * mask, x_rec * mask, pixel_list=None)
        return loss
