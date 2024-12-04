#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2.engine import (
    AMPTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
    SimpleTrainer,
    SSATTrainer
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")

import warnings
warnings.filterwarnings("ignore")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    
    kwargs = {} # for checkpointing
    # import pdb; pdb.set_trace()
    if cfg.train.amp.enabled:
        trainer = AMPTrainer(model, train_loader, optim)
    elif cfg.ssat.enabled:
        auxi_model = instantiate(cfg.ssat.model) 
        auxi_model.to(cfg.train.device) # ssat model to device
        auxi_optimizer = instantiate(cfg.ssat.optimizer)
        auxi_scheduler = instantiate(cfg.ssat.scheduler)

        kwargs.setdefault("auxiliary_head", auxi_model)
        kwargs.setdefault("auxiliary_optimizer", auxi_optimizer)
        if getattr(auxi_scheduler, "state_dict", None) is not None:
            kwargs.setdefault("auxiliary_lr_scheduler", auxi_scheduler)

        # register
        if not hasattr(model, "patch_size") and cfg.ssat.method == "simMIM":
            model.patch_size = cfg.ssat.patch_size
        if not hasattr(model, "in_chans") and cfg.ssat.method == "simMIM":
            model.in_chans = cfg.ssat.in_chans

        auxi_model = create_ddp_model(auxi_model, **cfg.train.ddp)
        trainer = SSATTrainer(model, auxi_model, train_loader, optim, auxi_optimizer, 
                              loss_scale=cfg.ssat.loss_scale, ssat_method=cfg.ssat.method, 
                              mask_ratio=cfg.ssat.mask_ratio, task_m=cfg.ssat.task_m)
    else:
        trainer = SimpleTrainer(model, train_loader, optim)

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        **kwargs,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.LRScheduler(optimizer=auxi_optimizer, scheduler=auxi_scheduler) if cfg.ssat.enabled else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.BestCheckpointer(checkpointer=checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

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


register_dentex_quadrant_dataset()

if __name__ == "__main__":
    invoke_main()  # pragma: no cover