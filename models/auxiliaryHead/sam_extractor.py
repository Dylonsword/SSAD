import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from models.SAM import sam_model_registry


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
        embedding = self.norm(self.globcnn(embedding).view(embedding.size(0), 1, embedding.size(1))).contiguous()

        embedding = embedding @ self.project
        return [embedding.view(embedding.size(0), embedding.size(2))]


class Tre:
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    x = torch.randn((4, 3, 512, 512))
    args = Tre()
    args.sam_checkpoint = ""
    # default, vit_b, vit_l, vit_h
    args.sam_size = "vit_b"
    extractor = SamExtractor(args)
    extractor.get_image_embedding(x)