import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from timm.models.vision_transformer import DropPath, Mlp


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x , attention_block=False):
        if attention_block == False:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        elif attention_block == True:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)           
            return attn

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x , attention_block = False):
        if attention_block == False:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
        elif attention_block == True:
            attention_matrix = self.attn(self.norm1(x),True)
            return attention_matrix 
        
        
class MAEHead(nn.Module):
    """Masked Autoencoder Decoder Head
    """
    def __init__(self, in_chans, patch_size, num_patches, embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, decoder_depth, norm_layer=nn.LayerNorm, pretrain=None, clip_criterion=None,
                 mask_ratio=0.6, output_rgb=False, use_cls_token=True, vis_mask_ratio=0):
        super(MAEHead, self).__init__()
        self.clip_criterion = clip_criterion
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.output_rgb = output_rgb
        self.mask_ratio = mask_ratio

        # encoder pos embed
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        # init pos embed
        encoder_pos_embed = self.get_2d_sincos_pos_embed(self.encoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=use_cls_token)
        self.encoder_pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        # init pos embed
        decoder_pos_embed = self.get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=use_cls_token)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if output_rgb:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        else:
            self.decoder_embed_vanilla = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
            torch.nn.init.normal_(self.vis_mask_token, std=.02)

        if pretrain:
            print("maeHead loadding pretrain weights")
            self._load_pretrain(pretrain)

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
            def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
                """
                embed_dim: output dimension for each position
                pos: a list of positions to be encoded: size (M,)
                out: (M, D)
                """
                assert embed_dim % 2 == 0
                omega = np.arange(embed_dim // 2, dtype=np.float64)
                omega /= embed_dim / 2.
                omega = 1. / 10000**omega  # (D/2,)

                pos = pos.reshape(-1)  # (M,)
                out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

                emb_sin = np.sin(out) # (M, D/2)
                emb_cos = np.cos(out) # (M, D/2)

                emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
                return emb

            assert embed_dim % 2 == 0

            # use half of dimensions to encode grid_h
            emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
            emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

            emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
            return emb

    # --------------------------------------------------------
    # 2D sine-cosine position embedding
    # References:
    # Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
    # MoCo v3: https://github.com/facebookresearch/moco-v3
    # --------------------------------------------------------
    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed
 
    def _load_pretrain(self, weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        
        model_state_dict = self.state_dict()
        for key, value in model_state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            
            drop_keys = []
            if key in checkpoint:
                if value.size() == checkpoint[key].size():
                    model_state_dict[key] = checkpoint[key]
                else:
                    drop_keys.append(key)
            else:
                drop_keys.append(key)
        print(f"drop keys {drop_keys}")
        
        self.load_state_dict(model_state_dict, strict=False)
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    # --------------------------------------------------------
    # References:
    # https://github.com/pengzhiliang/MAE-pytorch/blob/main/modeling_pretrain.py
    # --------------------------------------------------------
    def random_masking(self, x, mask_ratio):
        B, num_patches, C = x.shape    # B, N, L
        num_mask = int(mask_ratio * num_patches)
        overall_mask = np.zeros([B, num_patches])
        for i in range(B):
            mask = np.hstack([
                np.zeros(num_patches-num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        overall_mask = overall_mask.to(x.device, non_blocking=True)
        x_masked = x[~overall_mask].reshape(B, -1, C)  # ~mask means visible
        return x_masked, overall_mask

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

    def vitdet_encode(self, encoder, x, mask_ratio):
        vit = encoder.backbone.net

        # [B, C, H, W] -> [B, N, C]
        ori_x = self.patchify(x.clone())

        x = vit.patch_embed(x)
        B, h, w, C = x.shape
        if vit.pos_embed is not None:
            x = x + self.get_abs_pos(
                vit.pos_embed, vit.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            if vit.pretrain_use_cls_token:
                self.token_start_idx = 1    # pos 
                self.token_split_idx = 0
            else:
                self.token_start_idx = 0
                self.token_split_idx = 0
        else:
            x = x.reshape(B, -1, C)
            if self.use_cls_token:
                self.token_start_idx = 1
                self.token_split_idx = 0
                x = x + self.encoder_pos_embed[:, 1:, :]
            else:
                self.token_start_idx = 0
                self.token_split_idx = 0
                x = x + self.encoder_pos_embed

        if x.dim() == 4:
            x_reshape = x.reshape(x.shape[0], -1, x.shape[3])

        # masking: length -> length * mask_ratio
        # x_masked, mask, ids_restore = self.random_masking(x.clone(), mask_ratio)
        x_masked, mask = self.random_masking(x_reshape.clone(), mask_ratio)

        # vitdet using the windows attention
        B, m_N, L = x_masked.size()
        if self.vis_mask_ratio > 0:
            if vit.pos_embed is not None:
                vis_mask_token = self.vis_mask_token + self.get_abs_pos(
                vit.pos_embed, vit.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            # [b, h, w, d] -> [b, hw, d]
            vis_mask_token = vis_mask_token.reshape(1, -1, L).expand(B, -1, -1)
            vis_mask_token = vis_mask_token[~mask].reshape(B, -1, L)
            
            D = x_masked.size(1)
            noise = torch.rand(B, D, device=x.device)
            ids_restore = torch.argsort(noise, dim=1)

            len_keep = int(D * (1 - self.vis_mask_ratio))
            vis_mask = torch.ones([B, D], device=x.device)
            vis_mask[:, :len_keep] = 0
            vis_mask = torch.gather(vis_mask, dim=1, index=ids_restore).unsqueeze(-1)

            x_masked = x_masked * (1. - vis_mask) + vis_mask_token * vis_mask
        
        sqrt_n = math.ceil(math.sqrt(m_N))
        H = W = sqrt_n
        pad_num = H*W - m_N
        if pad_num > 0:
            pad_zeros = torch.zeros((B, pad_num, L), requires_grad=False, device=x.device)
            x_masked = torch.cat([x_masked, pad_zeros], dim=1)
        x_masked = x_masked.reshape(B, H, W, L)

        # reshape to ori
        # x = x.reshape(B, h, w, L)
        for blk in vit.blocks:
            x = blk(x)
            x_masked = blk(x_masked)

        # reshape to tokens
        x_masked = x_masked.reshape(B, -1, L)
        x_masked = x_masked[:, :m_N, :].contiguous()  # drop the pad
        x = x.reshape(B, -1, L)
        return x_masked, mask, x, ori_x
    
    def forward_encoder(self, encoder, x, mask_ratio, task_m):
        if task_m == "vitdet":
            return self.vitdet_encode(encoder, x, mask_ratio)
    
    # --------------------------------------------------------
    # References:
    # https://github.com/pengzhiliang/MAE-pytorch/blob/main/modeling_pretrain.py
    # --------------------------------------------------------
    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)

        B, N, C = x.shape
        detach_pos_embed = self.decoder_pos_embed.expand(B, -1, -1).clone().detach()
        if self.token_start_idx in (1, 2):
            cls_token = x[:, :self.token_split_idx, :]
            x_pure = x[:, self.token_split_idx:, :]
        elif self.token_start_idx == 0:
            x_pure = x

        wocls_detach_pos_embed = detach_pos_embed[:, self.token_start_idx:, :]
        pos_emd_vis = wocls_detach_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = wocls_detach_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_pure + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        
        # add cls token, if possible
        if self.token_start_idx in (1, 2):
            x_full = torch.cat([cls_token, x_full], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x_full)
        x = self.decoder_norm(x_full)

        # remove cls token
        if self.token_start_idx in (1, 2):
            x = x[:, self.token_split_idx:, :]

        return x
    
    def forward_loss(self, img, img_rec, mask, pixel_list=None):
        """
        Args:
            mask: (B, N)
        """
        loss_recon = F.mse_loss(img_rec, img, reduction='none')
        l2_loss = (loss_recon * mask[..., None]).sum() / (mask.sum() + 1e-5)
        clip_dist_loss = 0
        if self.clip_criterion is not None and self.output_rgb:
            # masked input
            rgb_img = self.unpatchify(img * mask[..., None])
            rgb_img_rec = self.unpatchify(img_rec * mask[..., None])
            clip_dist_loss += self.clip_criterion(rgb_img, rgb_img_rec, pixel_list)
        return l2_loss, clip_dist_loss
            
    def forward(self, imgs, encoder, masks=None, mask_ratio=0.6, task_m="vitdet"):
        if not isinstance(imgs, torch.Tensor):
            try:
                imgs = imgs.tensors
            except Exception:
                imgs = imgs.tensor

        # import pdb;pdb.set_trace()
        # This has the class token appended to it
        latent, mask, x_encoder, ori_x = self.forward_encoder(encoder, imgs, mask_ratio, task_m)
        
        #This doesnt have class token
        pred_decoder = self.forward_decoder(latent, mask)
        
        if self.output_rgb: # supervise at
            pred_decoder = self.decoder_pred(pred_decoder)  # B, N, Patch**2*3
            x_encoder = ori_x
        else:   # 
            x_encoder = self.decoder_embed_vanilla(x_encoder)   # representation project

        l2_loss, clip_dist_loss = self.forward_loss(x_encoder, pred_decoder, mask, [encoder.pixel_mean, encoder.pixel_std])
        return l2_loss + clip_dist_loss