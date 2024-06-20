# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 4
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = "relu"
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = "B"  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = False
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = (
        [96, 10000],
        [96, 10000],
        [64, 10000],
        [64, 10000],
        [64, 10000],
        [0, 10000],
        [0, 10000],
        [0, 256],
        [0, 256],
        [0, 192],
        [0, 192],
        [0, 96],
        [0, 10000],
    )

    # train param
    cfg.INPUT.FIX_SIZE = CN()
    cfg.INPUT.FIX_SIZE.FLAGS = True
    cfg.INPUT.FIX_SIZE.SIZE = (512, 512)

    # params of auxiliary head
    cfg.MODEL.AUXI = CN()
    cfg.MODEL.AUXI.FLAGS = False
    cfg.MODEL.AUXI.SIZE = 512
    cfg.MODEL.AUXI.MASK_PATCH_SIZE=32
    cfg.MODEL.AUXI.MODEL_PATCH_SIZE=4
    cfg.MODEL.AUXI.MASK_RATIO=0.6
    cfg.MODEL.AUXI.LR = 1e-4
    cfg.MODEL.AUXI.DECAY=0.05
    cfg.MODEL.AUXI.MOMENTUM=0.9
    cfg.MODEL.AUXI.HEAVILY=False
    cfg.MODEL.AUXI.RESTRUC_INDICES=[1,3]
    cfg.MODEL.AUXI.ENCODER_STRIDE=[4, 8, 16, 32]
    cfg.MODEL.AUXI.LOSS_SCALE=0.1  # the weights of contribution in training.
    cfg.MODEL.AUXI.PROJECT_NUM=[48, 192, 768, 3072]
    cfg.MODEL.AUXI.LOSS_TYPE="l1"    # l1, clip
    cfg.MODEL.AUXI.PRETRAIN=''
    cfg.MODEL.AUXI.METHOD= "simMIM"
    cfg.MODEL.AUXI.TASK_M= "diffdet"
    cfg.MODEL.AUXI.CLIP = CN()
    cfg.MODEL.AUXI.CLIP.MODEL_NAME="ViT-B/32"
    cfg.MODEL.AUXI.CLIP.AFFINE_TRANSFORM_FILL=1
    cfg.MODEL.AUXI.CLIP.N_AUG=4
    cfg.MODEL.AUXI.CLIP.STRUC_LAMBDA=0.3
    cfg.MODEL.AUXI.CLIP.DISTRI_LAMBDA=0.45  # TC scale
    cfg.MODEL.AUXI.SAM = CN()
    cfg.MODEL.AUXI.SAM.SIZE="vit_b"
    cfg.MODEL.AUXI.SAM.NECK=False
    cfg.MODEL.AUXI.SAM.CHECKPOINT="/data/xinyuan/tooth_disease_detection/3dSeg/segment-anything/sam_vit_b_01ec64.pth"
    cfg.MODEL.AUXI.SAM.PIXEL_LIST=True
    cfg.MODEL.AUXI.SAM.STRUC_LAMBDA=0.3
    cfg.MODEL.AUXI.SAM.DISTRI_LAMBDA=0.45  # TC scale