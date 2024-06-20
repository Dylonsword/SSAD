# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import time
import sys
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    create_ddp_model,
    AMPTrainer,
    SSATTrainer,
    SimpleTrainer,
    hooks,
)
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from models.auxiliaryHead import MIMHead, ClipExtractor, SamExtractor, CLIPLoss, get_params_group
from models.diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from models.diffusiondet.util.model_ema import (
    add_model_ema_configs,
    may_build_model_ema,
    may_get_ema_checkpointer,
    EMAHook,
    apply_model_ema_and_restore,
    EMADetectionCheckpointer,
)


class conf:
    def __init__(self) -> None:
        pass

class Trainer(DefaultTrainer):
    """Extension of the Trainer class adapted to DiffusionDet."""

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        auxi_model, auxi_optimizer, auxi_model_without_ddp = self.build_auxi(cfg, model)
        self.auxi_optimizer = auxi_optimizer
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        
        model = create_ddp_model(model, broadcast_buffers=False)
        if cfg.MODEL.AUXI.FLAGS:
            self._trainer = SSATTrainer(model, auxi_model, data_loader, optimizer, auxi_optimizer, cfg.MODEL.AUXI.LOSS_SCALE,
                                        ssat_method=cfg.MODEL.AUXI.METHOD, task_m=cfg.MODEL.AUXI.TASK_M)
            self.auxi_scheduler = self.build_lr_scheduler(cfg, auxi_optimizer)
        else:
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        
        ########## EMA ############
        kwargs = {
            "trainer": weakref.proxy(self),
        }
        if cfg.MODEL.AUXI.FLAGS:
            kwargs.setdefault("auxiliary_head", auxi_model_without_ddp)
            kwargs.setdefault("auxiliary_optimizer", auxi_optimizer)
            kwargs.setdefault("auxiliary_lr_scheduler", self.auxi_scheduler)

        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model
    
    @classmethod
    def build_auxi(cls, cfg, model):
        """Returns image reconstruction branch.
            torch.nn.Module, 
            optimizer,
            
        """
        if not cfg.MODEL.AUXI.FLAGS:    # if enable auxi head
            return None, None, None
        
        if cfg.MODEL.AUXI.LOSS_TYPE == "clip":
            args = dict(clip_model_name=cfg.MODEL.AUXI.CLIP.MODEL_NAME,
                        clip_affine_transform_fill=cfg.MODEL.AUXI.CLIP.AFFINE_TRANSFORM_FILL,
                        n_aug=cfg.MODEL.AUXI.CLIP.N_AUG,
                        struc_lambda=cfg.MODEL.AUXI.CLIP.STRUC_LAMBDA,
                        distri_lambda=cfg.MODEL.AUXI.CLIP.DISTRI_LAMBDA)
            clip_extractor = ClipExtractor(args)
            clip_criterion = CLIPLoss(args, clip_extractor)
        elif cfg.MODEL.AUXI.LOSS_TYPE == "sam":
            args = dict(sam_size=cfg.MODEL.AUXI.SAM.SIZE,
                        sam_checkpoint=cfg.MODEL.AUXI.SAM.CHECKPOINT,
                        neck=cfg.MODEL.AUXI.SAM.NECK,
                        struc_lambda=cfg.MODEL.AUXI.CLIP.STRUC_LAMBDA,
                        distri_lambda=cfg.MODEL.AUXI.CLIP.DISTRI_LAMBDA)
            params = conf()
            for k,v in args.items():
                setattr(params, k, v)

            clip_extractor = SamExtractor(params)
            clip_criterion = CLIPLoss(params, clip_extractor, using_pixel_list=cfg.MODEL.AUXI.SAM.PIXEL_LIST)
        else:
            clip_criterion = None

        encoder_stride = cfg.MODEL.AUXI.ENCODER_STRIDE
        restruc_indices = cfg.MODEL.AUXI.RESTRUC_INDICES
        num_feature = model.backbone.bottom_up.num_features
        embed_dims = {'T': 96,
                        'S': 96,
                        'B': 128,
                        'B-22k': 128,
                        'B-22k-384': 128,
                        'L-22k': 192,
                        'L-22k-384': 192,}
        assert cfg.MODEL.BACKBONE.NAME == "build_swintransformer_fpn_backbone", "auxiliary only supports Transformer-Base Backbone"
        embed_dim = embed_dims[cfg.MODEL.SWIN.SIZE]
        auxi_model = MIMHead(num_features=num_feature, restruc_indices=restruc_indices, 
                             encoder_stride=encoder_stride, embed_dim=embed_dim, heavily=cfg.MODEL.AUXI.HEAVILY,
                             clip_criterion=clip_criterion, 
                             pretrained_checkpoint=cfg.MODEL.AUXI.PRETRAIN)
        auxi_model = auxi_model.to(torch.device(cfg.MODEL.DEVICE))
        params_group = get_params_group(auxi_model)
        # optimizer
        lr = cfg.MODEL.AUXI.LR
        decay = cfg.MODEL.AUXI.DECAY
        momentum = cfg.MODEL.AUXI.MOMENTUM
        auxi_optimizer = torch.optim.SGD(params=params_group, lr=lr, momentum=momentum, nesterov=True, weight_decay=decay)
        auxi_model_without_ddp = auxi_model

        auxi_model = create_ddp_model(auxi_model, broadcast_buffers=False)
        return auxi_model, auxi_optimizer, auxi_model_without_ddp
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "lvis" in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.LRScheduler(self.auxi_optimizer, self.auxi_scheduler) if cfg.MODEL.AUXI.FLAGS else None,
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # save the best model checkpoint
            ret.append(hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                              checkpointer=self.checkpointer, val_metric="bbox/AP"))
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            # 1 epoch, wirte once
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.TEST.EVAL_PERIOD))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = args.output_dir
    if output_dir is None:
        if sys.gettrace() is not None:  # debug mode
            output_dir = f'./output_debug_{time.strftime("%m-%d_%H-%M", time.localtime())}'
        else:
            output_dir = f'./output_{time.strftime("%m-%d_%H-%M", time.localtime())}'

    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def register_dentex_quadrant_dataset():
    register_coco_instances(
        "dentex_quadrant_train",
        {},
        "dentex_dataset/coco/quadrant/annotations/instances_train2017.json",
        "dentex_dataset/coco/quadrant/train2017",
    )
    register_coco_instances(
        "dentex_quadrant_val",
        {},
        "dentex_dataset/coco/quadrant/annotations/instances_val2017.json",
        "dentex_dataset/coco/quadrant/val2017",
    )
    # enu32
    register_coco_instances(
        "dentex_enumeration32_train",
        {},
        "dentex_dataset/coco/enumeration32/annotations/instances_train2017.json",
        "dentex_dataset/coco/enumeration32/train2017",
    )
    register_coco_instances(
        "dentex_enumeration32_val",
        {},
        "dentex_dataset/coco/enumeration32/annotations/instances_val2017.json",
        "dentex_dataset/coco/enumeration32/val2017",
    )
    # diease
    register_coco_instances(
        "dentex_disease_train",
        {},
        "dentex_dataset/coco/disease/annotations/instances_train2017.json",
        "dentex_dataset/coco/disease/train2017",
    )
    register_coco_instances(
        "dentex_disease_val",
        {},
        "dentex_dataset/coco/disease/annotations/instances_val2017.json",
        "dentex_dataset/coco/diseases/val2017",
    )


register_dentex_quadrant_dataset()  # call out of __main__ for multi-processing

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--output-dir", default=None, type=str, help="path to save output results")
    args = parser.parse_args()
    # register_dentex_quadrant_dataset()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
