
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

from .fcos_fpn import model

# another params from file "base.py"
base = model_zoo.get_config("common/base.py")
# component
train = base.train
# redefine train params
train.output_dir = "train_out/chest/fcos/base-res50"
dataloader = base.dataloader
optimizer = base.optimizer
lr_multiplier = base.lr_multiplier
# ssat
ssat = base.ssat

optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 2
train.init_checkpoint = "checkpoints/R-50.pkl"