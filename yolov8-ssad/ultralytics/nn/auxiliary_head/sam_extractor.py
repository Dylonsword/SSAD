import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# import sys
# sys.path.append("/data/xinyuan/tooth_disease_detection/det/DentexSegAndDet/")
from SAM import sam_model_registry


class Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-8, max=None) # normalize to [0, 1], (N, 3, H, W)
        return x


class SamExtractor(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.neck = getattr(args, "neck", True)
        self.model = sam_model_registry[args.sam_size](getattr(args, "sam_checkpoint"))
        self.sam_input_size = 224    # 224 for smaller memory desire

        self.sam_transforms = T.Compose([
            T.Resize(self.sam_input_size, max_size=384, interpolation=InterpolationMode.BICUBIC),
            T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        ])
        # self.GlobalPool = nn.AdaptiveAvgPool2d(1)
        # linear_mapping
        in_feat = 768 if not self.neck else 256
        self.globcnn = nn.Conv2d(in_channels=in_feat, out_channels=in_feat, 
                                 kernel_size=14)
        self.norm = nn.LayerNorm(in_feat, eps=1e-6)
        self.project = nn.Parameter(torch.rand((1, in_feat, 512)), requires_grad=True)

    def get_image_embedding(self, x, aug=True):
        views = self.sam_transforms(x)
        # input [b, 3, 512, 512]
        # backbone: [b,768,32,32], neck: [b, 256, 32, 32]
        embedding = self.model(views, neck=self.neck)
        # Try add GlobalPool
        # embedding = self.GlobalPool(embedding).view(embedding.size(0), embedding.size(1))  # B, C, 1, 1 -> B, C
        embedding = self.norm(self.globcnn(embedding).view(embedding.size(0), 1, embedding.size(1))).contiguous()

        embedding = embedding @ self.project
        return [embedding.view(embedding.size(0), embedding.size(2))]


class Tre:
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    x = torch.randn((4, 3, 512, 512))
    args = Tre()
    args.sam_checkpoint = "/data/xinyuan/tooth_disease_detection/3dSeg/segment-anything/sam_vit_b_01ec64.pth"
    # default, vit_b, vit_l, vit_h
    args.sam_size = "vit_b"
    # extractor = MedSamExtractor(args)
    extractor.get_image_embedding(x)