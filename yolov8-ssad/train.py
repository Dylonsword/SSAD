import os
from ultralytics.models import YOLO
os.environ["WANDB_DISABLED"] = "true"

yolo_model = YOLO(model="./yolov8l.yaml", task="detect").load("./yolov8l.pt")

# Train the model
train_results = yolo_model.train(
    cfg="/data/xinyuan/tooth_disease_detection/yolov8-ssad/hyp.baseline.yaml",
    data="/data/xinyuan/tooth_disease_detection/yolov8-ssad/dentex.yaml",  # path to dataset YAML
    batch=4,
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="6",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    exist_ok=True,
    name="v8l-baseline-6-true",
    ssad_flag=False,
    ssad_mask_patch_size=32,
    ssad_model_patch_size=2,  # stem layer downsample 2 factor
    ssad_mask_ratio=0.6,
    ssad_heavily=True,
    ssad_loss_scale=0.1,
)