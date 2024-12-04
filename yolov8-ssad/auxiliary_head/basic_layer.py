import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional

from timm.models.layers import trunc_normal_


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class PatchUnmerging(nn.Module):
    """ Patch Unmerging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim // 4, dim // 4, bias=False)
        self.norm = norm_layer(dim // 4)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, 4*C).
            H, W: Spatial resolution of the input feature.
        """
        B, _, C = x.shape
        x = x.reshape(B, H, W, C)
        x = self.pixel_shuffle(x)
        x = self.linear(x)
        x = self.norm(x)
        
        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: nn.Module = nn.GELU
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act = act_layer()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class RGBBlock(nn.Module):
    def __init__(self, in_c, upsample, rgba=False) -> None:
        super(RGBBlock, self).__init__()
        out_c = 3 if not rgba else 4
        self.conv = conv1x1(in_c, out_c)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        ) if upsample else None

    def forward(self, x, prev_rgb=None):
        x = self.conv(x)
        if prev_rgb is not None:
            x = x + prev_rgb
        
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ConvDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, upsample=True, to_rgb=True, upsample_rgb=False, rgba=False, wnorm=False, bias=True) -> None:
        super(ConvDecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.conv1 = conv3x3(in_c, out_c, bias=bias)
        self.conv2 = conv3x3(out_c, out_c, bias=bias)
        if wnorm:
            self.norm1 = nn.InstanceNorm2d(out_c, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_c, affine=True)
        else:
            self.norm1 = None
            self.norm2 = None
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.to_rgb = RGBBlock(out_c, upsample=upsample_rgb, rgba=rgba) if to_rgb else None

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, prev_rgb=None):
        if self.upsample is not None:
            x = self.upsample(x)

        x = self.conv1(x)
        x = self.act(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        # may add skip

        x = self.conv2(x)
        x = self.act(x)
        if self.norm2 is not None:
            x = self.norm2(x)

        rgb = self.to_rgb(x, prev_rgb)
        return x, rgb


class ConvDecoder(nn.Module):
    def __init__(self, num_features, w_norm=False, bias=True) -> None:
        super(ConvDecoder, self).__init__()

        convBlocks = []
        for i, (in_c, out_c) in enumerate(zip(num_features[:-1], num_features[1:])):
            convBlocks.append(
                ConvDecoderBlock(in_c, out_c, upsample=True, to_rgb=True, upsample_rgb=True, rgba=False, wnorm=w_norm, bias=bias)
                )

        self.convBlocks = nn.ModuleList(convBlocks)
    
    def forward(self, x, upsamle_first=False):
        if upsamle_first:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        rgb = None
        for block in self.convBlocks:
            x, rgb = block(x, rgb)
        return rgb