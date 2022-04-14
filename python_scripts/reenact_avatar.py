import torch

from nha.data.real import RealDataModule, tracking_results_2_data_batch
from nha.models.nha_optimizer import NHAOptimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from nha.util.general import *
import matplotlib.pyplot as plt
from configargparse import ArgumentParser as ConfigArgumentParser
import os
from pathlib import Path


@torch.no_grad()
def reenact_avatar(target_model: NHAOptimizer, driving_model: NHAOptimizer, target_tracking_results: dict,
                   driving_tracking_results: dict, outpath: Path,
                   neutral_driving_frame=0, neutral_target_frame=0,
                   batch_size=3, plot=False):
    base_drive_sample = dict_2_device(tracking_results_2_data_batch(driving_tracking_results, [neutral_driving_frame]),
                                      driving_model.device)
    base_target_sample = dict_2_device(tracking_results_2_data_batch(target_tracking_results, [neutral_target_frame]),
                                       target_model.device)

    base_drive_params = driving_model._create_flame_param_batch(base_drive_sample)
    base_target_params = target_model._create_flame_param_batch(base_target_sample)

    tmp_dir_pred = Path("/tmp/scene_reenactment_pred")
    tmp_dir_drive = Path("/tmp/scene_reenactment_drive")
    os.makedirs(tmp_dir_drive, exist_ok=True)
    os.makedirs(tmp_dir_pred, exist_ok=True)
    os.makedirs(outpath.parent, exist_ok=True)
    os.system(f"rm -r {tmp_dir_drive}/*")
    os.system(f"rm -r {tmp_dir_pred}/*")
    frameid2imgname = lambda x: f"{x:04d}.png"

    for idcs in tqdm(torch.split(torch.from_numpy(driving_tracking_results["frame"]), batch_size)):
        batch = dict_2_device(tracking_results_2_data_batch(driving_tracking_results, idcs.tolist()), target_model.device)

        rgb_driving = driving_model.forward(batch, symmetric_rgb_range=False)[:, :3].clamp(0,1)

        # change camera parameters
        batch["cam_intrinsic"] = base_target_sample["cam_intrinsic"].expand_as(batch["cam_intrinsic"])
        batch["cam_extrinsic"] = base_target_sample["cam_extrinsic"].expand_as(batch["cam_extrinsic"])

        rgb_target = target_model.predict_reenaction(batch, driving_model=driving_model,
                                                     base_target_params=base_target_params,
                                                     base_driving_params=base_drive_params)

        for frame_idx, pred, drive in zip(batch["frame"], rgb_target, rgb_driving):
            save_torch_img(pred, tmp_dir_pred / frameid2imgname(frame_idx.cpu().item()))
            save_torch_img(drive, tmp_dir_drive / frameid2imgname(frame_idx.cpu().item()))

            if plot:
                fig, axes = plt.subplots(ncols=2)
                axes[0].imshow(drive.cpu().permute(1, 2, 0))
                axes[1].imshow(pred.cpu().permute(1, 2, 0))
                plt.show()
                plt.close()

    os.makedirs(outpath, exist_ok=True)
    os.system(f"ffmpeg -pattern_type glob -i {tmp_dir_pred}/'*.png' -c:v libx264 -preset slow  -profile:v high "
              f"-level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac {outpath}/Reenactment_pred.mp4 -y")
    os.system(f"ffmpeg -pattern_type glob -i {tmp_dir_drive}/'*.png' -c:v libx264 -preset slow  -profile:v high "
              f"-level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac {outpath}/Reenactment_drive.mp4 -y")
    os.system(f"ffmpeg  -i {outpath}/Reenactment_drive.mp4 -i {outpath}/Reenactment_pred.mp4 "
              f"-filter_complex hstack=inputs=2 {outpath}/Reenactment_combined.mp4 -y")



if __name__ == "__main__":
    parser = ConfigArgumentParser()
    parser.add_argument('--config', required=False, is_config_file=True)
    parser.add_argument("--driving_ckpt", type=Path, required=True)
    parser.add_argument("--target_ckpt", type=Path, required=True)
    parser.add_argument("--driving_tracking_results", type=Path, required=True)
    parser.add_argument("--target_tracking_results", type=Path, required=True)
    parser.add_argument("--neutral_driving_frame", type=int, required=False, default=0)
    parser.add_argument("--neutral_target_frame", type=int, required=False, default=0)
    parser.add_argument("--outpath", type=Path, required=True)

    args = parser.parse_args()

    driving_model = NHAOptimizer.load_from_checkpoint(args.driving_ckpt).cuda().eval()
    target_model = NHAOptimizer.load_from_checkpoint(args.target_ckpt).cuda().eval()

    reenact_avatar(target_model, driving_model,
                   target_tracking_results=np.load(args.target_tracking_results),
                   driving_tracking_results=np.load(args.driving_tracking_results),
                   outpath=args.outpath,
                   neutral_driving_frame=args.neutral_driving_frame,
                   neutral_target_frame=args.neutral_target_frame)
