import json
import time
from collections import OrderedDict

from torch.utils.data import DataLoader

from nha.evaluation.eval_suite import evaluate_models
from nha.util.log import get_logger
import os

import pytorch_lightning as pl
from pathlib import Path
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from configargparse import ArgumentParser as ConfigArgumentParser

from nha.evaluation.visualizations import generate_novel_view_folder, reconstruct_sequence

logger = get_logger(__name__)


def train_pl_module(optimizer_module, data_module, args=None):
    """
    optimizes an instance of the given optimization module on an instance of a given data_module. Takes arguments
    either from CLI or from 'args'

    :param optimizer_module:
    :param data_module:
    :param args: list similar to sys.argv to manually insert args to parse from
    :return:
    """
    # creating argument parser
    parser = ArgumentParser()
    parser = optimizer_module.add_argparse_args(parser)
    parser = data_module.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument('--config', required=True, is_config_file=True)
    parser.add_argument("--checkpoint_file", type=str, required=False, default="",
                        help="checkpoint to load model from")

    args = parser.parse_args() if args is None else parser.parse_args(args)
    args_dict = vars(args)

    print(f"Start Model training with the following configuration: \n {parser.format_values()}")

    # init datamodule
    while True:
        try:
            data = data_module(**args_dict)
            data.setup()
            break
        except FileNotFoundError as e1:
            print(e1)
            print("Retry data loading after 2 minutes ... zZZZZ")
            time.sleep(2 * 60)

    # init optimizer
    args_dict['max_frame_id'] = data.max_frame_id

    if args.checkpoint_file:
        model = optimizer_module.load_from_checkpoint(args.checkpoint_file, strict=True, **args_dict)
    else:
        model = optimizer_module(**args_dict)

    stages = ["offset", "texture", "joint"]
    stage_jumps = [args_dict["epochs_offset"], args_dict["epochs_offset"] + args_dict["epochs_texture"],
                   args_dict["epochs_offset"] + args_dict["epochs_texture"] + args_dict["epochs_joint"]]

    experiment_logger = TensorBoardLogger(args_dict["default_root_dir"],
                                          name="lightning_logs")
    log_dir = Path(experiment_logger.log_dir)

    for i, stage in enumerate(stages):
        current_epoch = torch.load(args_dict["checkpoint_file"])["epoch"] if args_dict["checkpoint_file"] else 0
        if current_epoch < stage_jumps[i]:
            logger.info(f"Running the {stage}-optimization stage.")

            ckpt_file = args_dict["checkpoint_file"] if args_dict["checkpoint_file"] else None
            trainer = pl.Trainer.from_argparse_args(args, callbacks=model.callbacks,
                                                    resume_from_checkpoint=ckpt_file,
                                                    max_epochs=stage_jumps[i],
                                                    logger=experiment_logger)

            # training
            trainer.fit(model,
                        train_dataloader=data.train_dataloader(batch_size=data._train_batch[i]),
                        val_dataloaders=data.val_dataloader(batch_size=data._val_batch[i]))

            ckpt_path = Path(trainer.log_dir) / "checkpoints" / (stage + "_optim.ckpt")
            trainer.save_checkpoint(ckpt_path)
            ckpt_path = Path(trainer.log_dir) / "checkpoints" / "last.ckpt"
            args_dict["checkpoint_file"] = ckpt_path

    # visualizations and evaluations
    proc_id = int(os.environ.get("SLURM_PROCID", 0))
    if proc_id == 0:
        model_name = log_dir.name
        logger.info("Producing Visualizations")
        vis_path = log_dir / "NovelViewSynthesisResults"
        model = optimizer_module.load_from_checkpoint(args_dict["checkpoint_file"],
                                                      strict=True, **args_dict).eval().cuda()
        generate_novel_view_folder(model, data, angles=[[0, 0], [-30, 0], [-60, 0]],
                                   outdir=vis_path, center_novel_views=True)
        # os.system("module load FFmpeg")
        os.system(
            f"for split in train val; do for angle in 0_0 -30_0 -60_0; do ffmpeg -pattern_type glob -i {vis_path}/$split/$angle/'*.png' {vis_path}/{vis_path.parent.name}-$split-$angle.mp4;done;done")

        # freeing up space
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass

        # quantitative evaluation of val dataset
        bs = max(args_dict["validation_batch_size"])
        dataloader = DataLoader(data._val_set, batch_size=bs,
                                num_workers=bs, shuffle=False)
        model_dict = {model_name: args_dict["checkpoint_file"]}
        model_dict = OrderedDict(model_dict)
        eval_path = log_dir / f"QuantitativeEvaluation-{model_name}.json"

        eval_dict = evaluate_models(models=model_dict, dataloader=dataloader)
        with open(eval_path, "w") as f:
            json.dump(eval_dict, f)

        # scene reconstruction
        reconstruct_sequence(model_dict, dataset=data._val_set, batch_size=bs,
                             savepath=str(log_dir / f"SceneReconstruction{model_name}-val.mp4"))
        reconstruct_sequence(model_dict, dataset=data._train_set, batch_size=bs,
                             savepath=str(log_dir / f"SceneReconstruction-{model_name}-train.mp4"))
