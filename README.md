# SSAD: Self-supervised Auxiliary Detection Framework for Panoramic X-ray based Dental Disease Diagnosis
This is the offical implement code of artical [https://arxiv.org/abs/2406.13963](https://arxiv.org/abs/2406.13963). It employs the reconstruction-based Self-supervised methods to assist deep-learning model in diagnosing the Dental Disease, and the whole framework as below:
![SSAD Framework](https://github.com/Dylonsword/SSAD/blob/main/figure/ssad_framework.png)

## Requirements
```
# STEP1: Find a PyTorch version that matches your CUDA version from the official PyTorch website, for example:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# STEP2: Install Requirements
pip install -r requirements.txt

# STEP3 Install CLIP
cd models/CLIP && pip install .

# STEP4 Refer to DINO for guidance on installing MultiScaleDeformableAttention.
```
DINO website is [there](https://github.com/IDEA-Research/DINO)


## Train
**Data Preparation**: Please refer at [there](https://github.com/xyzlancehe/DentexSegAndDet)
```
## Train SSAD with detectron2, For Example:
# yaml
python train_diffdet.py --num-gpus 1 --config-file configs/faster_rcnn/diffdet.dentex.swinbase.disease.yaml OUTPUT_DIR "the path of model weights" MODEL.WEIGHTS "checkpoints/swin_base_patch4_window7_224_22k.pth"

# py
python train_lazyconfig_net.py --num-gpus 1 --config-file configs/faster_rcnn/faster_rcnn_vitB_vitdet.py "train.output_dir='the path of model weights'" "train.init_checkpoint='checkpoints/vitdet.pkl'" 

## Train Yolov8, For Example:
cd yolov8-ssad
# Run Baseline
export DEFAULT_CFG_PATH=./hyp.baseline.yaml && python train.py
# Run SSAD
export DEFAULT_CFG_PATH=./hyp.ssad.yaml && python train_ssad.py
```

## Eval


## Model Zoo
| Network | Encoder | SSAD | checkpoint | AP50:95 |
|--------|--------|--------|--------|--------|
| DINO | Res50 | N |  | 15.42 |
| DINO | Res50 | Y |  | 17.12 |
| YOLOv8-L  | CSPDarknet | N |  |  33.9 |
| YOLOv8-L  | CSPDarknet | Y |  | 37.0 |
| Faster RCNN | ViTDet-B | N |  | 28.93 |
| Faster RCNN | ViTDet-B | Y |  | 31.04 |
| FCOS | ViTDet-B | N |  | 14.46 |
| FCOS | ViTDet-B | Y |  | 19.18 |
| DiffusionDet | Swin-B | N |  | 10.25 |
| DiffusionDet | Swin-B | Y |  | 11.40 |
| HierarchicalDet | Swin-B | N |  | 24.15 |
| HierarchicalDet | Swin-B | Y |  | 26.70 |