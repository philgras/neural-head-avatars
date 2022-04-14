from nha.data.real import digitize_segmap
from itertools import product
import sys
from nha.util.general import *
from matplotlib.animation import FuncAnimation
import os

from nha.models.nha_optimizer import NHAOptimizer
from nha.data.real import RealDataModule
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from torch.utils.data.dataloader import DataLoader
from nha.util.general import dict_2_device, fill_tensor_background
import torch
import numpy as np
from torchvision.transforms import ToPILImage
from tqdm import tqdm

topil = ToPILImage()

sys.path.append("./deps")
sys.path.append("./deps/flame_fitting")

logger = get_logger(__name__)


@torch.no_grad()
def reconstruct_sequence(models,
                         dataset, fps=25, batch_size=10, savepath=None):
    """
    Produces a reconstructed animation of a tracked sequence using the tracking results provided by dataset and the
    models provided in models
    :param dataset: Dataset instance that provides tracking results and ground truth data
    :param fps:
    :param batch_size:
    :param models: Ordered dict with structure:
                    key = model names
                    val = dict(ckpt="/path/to/checkpoint/file", type="str_of_model_type")
    :param savepath: Path to save the animation to
    :return:
    """
    figsize = 5

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # MODEL INFERENCE
    img_dict = dict()
    for m_name, m_ckpt in models.items():
        logger.info(f"Start Inference for model {m_name}")
        hparams_file = Path(m_ckpt).parents[1] / "hparams.yaml"
        m = NHAOptimizer.load_from_checkpoint(m_ckpt,
                                              hparams_file=str(hparams_file)).cuda()

        m.eval()

        pred_rgb = []
        pred_geom = []
        pred_seg = []
        for batch in tqdm(iterable=dataloader):
            batch = dict_2_device(batch, "cuda")
            rgba_pred = m.forward(dict_2_device(batch, "cuda"), symmetric_rgb_range=False)
            pred_rgb.append(rgba_pred[:,:3].cpu())
            pred_seg.append(rgba_pred[:,3:].cpu())
            pred_geom.append(m.predict_shaded_mesh(dict_2_device(batch, "cuda")).cpu()[:,:3])
        logger.info(f"Finished Inference for model {m_name}")
        pred_rgb = torch.cat(pred_rgb, dim=0)
        pred_geom = torch.cat(pred_geom, dim=0)
        pred_seg = torch.cat(pred_seg, dim=0)
        img_dict[m_name] = dict(rgb=pred_rgb, geom=pred_geom, seg=pred_seg)
    logger.info("Finished Model Inference")

    # MAKE ANIMATION
    logger.info("Start Plotting")

    titles = ["GT", "PRED", "DIST", "GEOM"]

    # init animation
    N = len(titles)
    fig, axes = plt.subplots(ncols=N, nrows=len(models),
                             figsize=(N * figsize, figsize * len(img_dict)))
    actors = []
    axes = axes.reshape(len(models), N)
    for i, j in product(range(len(models)), range(N)):
        actors.append(axes[i, j].imshow(torch.zeros_like(pred_geom[0].permute(1, 2, 0).cpu())))
        axes[i, j].tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        if j == 0:
            axes[i, j].set_ylabel(list(models.keys())[i])
        if i == 0:
            axes[i, j].set_title(titles[j])
    plt.tight_layout()

    def update(n):
        sample = dataset[n]
        rgb_gt = fill_tensor_background(sample["rgb"][None], (sample["seg"][None] != 0).float())[
            0].permute(1, 2, 0)
        rgb_gt = rgb_gt * .5 + .5
        for k, m_name in enumerate(img_dict.keys()):
            seg_gt = digitize_segmap(sample["seg"][None])[0][0].cpu()
            # PLOT IMAGES
            geom_pred = img_dict[m_name]["geom"][n].permute(1, 2, 0).cpu()
            seg_pred = img_dict[m_name]["seg"][n][0].cpu()
            seg_diff = torch.zeros_like(rgb_gt)  # H x W x 3
            seg_diff[seg_pred.bool() & seg_gt.bool()] = torch.tensor([0., 1., 0.])
            seg_diff[seg_pred.bool() & (~seg_gt.bool())] = torch.tensor([1., 0., 0.])
            seg_diff[(~seg_pred.bool()) & seg_gt.bool()] = torch.tensor([0., 0., 1.])

            rgb_pred = img_dict[m_name]["rgb"][n].permute(1, 2, 0).cpu()
            rgb_dist = tensor_distance_img(rgb_pred.permute(2, 0, 1)[None],
                                           rgb_gt.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
            actors[k * N + 0].set_array(rgb_gt)
            actors[k * N + 1].set_array(rgb_pred)
            actors[k * N + 2].set_array(rgb_dist)
            actors[k * N + 3].set_array(geom_pred)

        return actors

    logger.info("Started Animation")
    anim = FuncAnimation(fig, func=update, frames=len(dataset), interval=1000. / fps)
    if savepath is not None:
        callback = lambda i, n: print(f'Saving frame {i} of {n}')
        os.makedirs(Path(savepath).parent, exist_ok=True)
        anim.save(savepath, progress_callback=callback)
    logger.info("Finished Animation")
    return anim


@torch.no_grad()
def generate_novel_view_folder(model: NHAOptimizer,
                               data_module: RealDataModule,
                               angles: list,
                               outdir: Path = None,
                               gt_outdir: Path = None,
                               center_novel_views=False,
                               split="all"):
    """
    generate novel view folder for quantitative and qualitative comparison with related work
    if outdir is not specified only saves gt information
    :param model:
    :param data_module:
    :param angles:
    :param outdir:
    :param gt_outdir: path to save gt files to
    :return:
    """

    model.eval()

    if split == "all":
        splits = ["train", "val"]
    else:
        splits = [split]

    for split in splits:
        print(f"Making Plots for {split} set")

        # setup output directory
        if outdir is not None:
            splitdir = outdir / split
            os.makedirs(splitdir, exist_ok=True)

        ds = data_module._train_set if split == "train" else data_module._val_set
        dataloader = DataLoader(ds, batch_size=3,  # model.hparams["validation_batch_size"],
                                num_workers=3,  # model.hparams["validation_batch_size"]
                                )

        if gt_outdir is not None:
            gt_angledir = gt_outdir / split / "0_0"
            os.makedirs(gt_angledir, exist_ok=True)

        for batch in tqdm(iterable=dataloader):
            batch = dict_2_device(batch, model.device)

            if outdir is not None:
                gt_pose = batch["flame_pose"][:, :3].cpu().numpy()
                for az, el in angles:
                    angledir = splitdir / f"{az}_{el}"
                    os.makedirs(angledir, exist_ok=True)

                    # finding correct rotation vectors:
                    rots = []
                    for r in gt_pose:
                        rot = R.from_rotvec([0, az * np.pi / 180., 0]) * R.from_rotvec([-el * np.pi / 180., 0, 0])
                        rot = (rot * R.from_rotvec(r)).as_rotvec()
                        rots.append(rot)
                    batch["flame_pose"][:, :3] = torch.tensor(np.stack(rots), device=model.device)
                    center_pred = center_novel_views and (az != 0 or el != 0)  # front view is never recentered
                    pred_imgs = model.forward(batch, center_prediction=center_pred, symmetric_rgb_range=False)

                    for img, frame in zip(pred_imgs, batch["frame"]):
                        f_outpath = angledir / f"{frame.item():04d}.png"
                        topil(img.detach().cpu()).save(f_outpath)

            # saving gt data
            if gt_outdir is not None:
                gt_imgs = fill_tensor_background(batch["rgb"], (batch["seg"] > 0).float()) * .5 + .5

                for img, frame in zip(gt_imgs, batch["frame"]):
                    f_outpath = gt_angledir / f"{frame.item():04d}.png"
                    topil(img.detach().cpu()).save(f_outpath)
