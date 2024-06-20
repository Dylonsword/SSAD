from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from models.diffusiondet.dataset_mapper import SimpleDatasetMapper


dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="dentex_disease_train"),
    mapper=L(SimpleDatasetMapper)(
        is_train=True,
        crop_gen=None, 
        tfm_gens=[
            L(T.RandomFlip)(),
            L(T.Resize)(shape=(512, 512))
        ], 
        image_format="RGB",
        mask_generator=None     # If using simMIM, must init an MaskGerator
    ),
    total_batch_size=16,
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="dentex_disease_val", filter_empty=False),
    mapper=L(SimpleDatasetMapper)(
        is_train=False,
        tfm_gens=[
            L(T.Resize)(shape=(512, 512))
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=8,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)