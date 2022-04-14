from nha.evaluation import metrics as nha_metrics
import torch.nn as nn
import nha.models
from nha.util.general import *
from nha.data.real import digitize_segmap

import torch
import torchvision.transforms.functional as ttF

from typing import *
from tqdm import tqdm
from collections import OrderedDict


class Evaluator:
    """
    class to conveniently calculate various scores for predicted images.
    Attention: Keep in mind to blur mask before doing alpha merging such
    that CPBD (sharpness evaluation) is not influenced by these edge artifacts!
    """

    def __init__(
        self,
        metrics=["L1", "L2", "PSNR", "MS_SSIM", "LMK", "LPIPS", "CPBD"],
        device="cuda",
        load_bbx_detector=False,
    ):
        """
        :param metrics: a list of metrics to include in the evaluation process
        :param device: where to run the evaluation
        """

        # ensures that if LMK in metrics, its calculated first such that bbs that were detected on the way can be reused
        self._metrics = OrderedDict()
        if "LMK" in [m.upper() for m in metrics]:
            metrics = list(metrics)
            metrics.remove("LMK")
            metrics = ["LMK"] + metrics

        self._device = device
        self._bbx_detector = (
            nha_metrics.FaceBBxDetector(device) if load_bbx_detector else None
        )

        for m in metrics:
            m = m.upper()
            if m == "LMK":
                self._metrics["LMK"] = nha_metrics.EuclLmkDistance(device)
            elif m == "L1":
                self._metrics["L1"] = nn.L1Loss(reduction="none")
            elif m == "L2":
                self._metrics["L2"] = nn.MSELoss(reduction="none")
            elif m == "MS_SSIM":
                self._metrics["MS_SSIM"] = nha_metrics.MS_SSIM(device)
            elif m == "PSNR":
                self._metrics["PSNR"] = nha_metrics.PSNR(device)
            elif m == "LPIPS":
                self._metrics["LPIPS"] = nha_metrics.LPIPS(device)
            elif m == "CPBD":
                self._metrics["CPBD"] = nha_metrics.CPBD(device)

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        reduction="none",
        bbs=None,
        gt_landmarks=None,
    ):
        """
        returns dictionary with evaluation scores for L1, L2, PSNR, MS_SSIM, LMK, LPIPS, CPBD.
        If reduction=="none", each dictionary value is a torch tensor of length N.
        If reduction in ["mean", "sum"], dict values are scalar tensors
        :param pred: torch tensor of shape N x 3 x H x W with entries -1 ... +1
        todo add predicted landmarks
        :param gt: torch.tensor of shape N x 3 x H x W  with entries -1 ... +1
        :param bbs: None or np array of shape N x 5
        :param gt_landmarks: None or np array of shape N x 68 x 2
        :param reduction: How to reduce scores along N dim. One of
                        - "none": no reduction, scores will have length N
                        - "mean": scalar scores
        :return:
        """

        scores = dict()
        pred = pred.to(self._device)
        gt = gt.to(self._device)

        for name, metric in self._metrics.items():
            if name in ["L1", "L2"]:
                scores[name] = torch.flatten(metric(pred, gt), start_dim=1).mean(dim=-1)
            elif name == "LMK":
                scores[name], bbs = metric(pred, gt, return_bbs=True, bbs=bbs)
            elif name == "CPBD":
                scores[name] = metric(pred)
            else:
                scores[name] = metric(pred, gt)

        if reduction == "mean":
            for key, val in scores.items():
                scores[key] = torch.mean(val)
        if reduction == "sum":
            for key, val in scores.items():
                scores[key] = torch.sum(val)

        return scores


@torch.no_grad()
def evaluate_models(
    models: OrderedDict,
    dataloader,
    metrics=["L1", "L2", "PSNR", "MS_SSIM", "LMK", "LPIPS", "CPBD"],
    blur_seg=0.0,
):
    """
    evaluates the performances of several models and creates a comparison dictionary. Keys are the same as given in
    'models'; vals are dictionaries with keys:
        - [Metric] ... contains evaluated metrics score as float

    :param models: dict where key are the model names, each value is another dict with keys
                    'ckpt' (path to checkpoint file) and 'type' str model type identifier as defined in
                    nha.util.models.__init__
                    e.g.: {"model_A": "/path/to/checkpoint"},
                           "model_B": ...,  }


    :param dataloader: validation dataloader
    :param blur_seg: specifies factor to blur segmentation masks by before doing bg filling. If 0 has no effect. May
                     be used for more useful cpbd score evaluation
    :param debug: if true: plots example of compared prediction - gt image pair
    :return: dict with following structure:
                {"MODEL_NAME":
                    {"SCORE_NAME": score value (as float),
                    }
                ...,
                }
    """

    evaluator = Evaluator(metrics=metrics)
    scores = OrderedDict()

    for model_name, ckpt in models.items():
        # loading model
        model = nha.models.nha_optimizer.NHAOptimizer.load_from_checkpoint(ckpt).cuda()
        model.eval()

        # evaluation:
        scores[model_name] = OrderedDict()
        for batch in tqdm(
            iterable=dataloader, desc=f"Quantitative Analysis of '{model_name}'"
        ):
            batch = dict_2_device(batch, model.device)

            # get prediction
            pred_rgba = model.forward(batch)
            pred_rgb, pred_mask = pred_rgba[:, :3], pred_rgba[:, 3:]

            # get gt
            gt_rgb = batch["rgb"]
            gt_mask = digitize_segmap(batch["seg"]).float()
            gt_rgb = fill_tensor_background(gt_rgb, gt_mask)

            # blur masks if specified:
            if blur_seg > 0:
                pred_mask = ttF.gaussian_blur(
                    pred_mask, 2 * int(2 * blur_seg) + 1, blur_seg
                )
                gt_mask = ttF.gaussian_blur(
                    gt_mask, 2 * int(2 * blur_seg) + 1, blur_seg
                )

                pred_rgb = fill_tensor_background(pred_rgb, pred_mask)
                gt_rgb = fill_tensor_background(gt_rgb, gt_mask)

            # evaluate image
            scores[model_name] = cat_torch_dicts(
                scores[model_name], evaluator(pred_rgb, gt_rgb)
            )

        # calculating average score for evaluation metrics
        for key, val in scores[model_name].items():
            scores[model_name][key] = val.mean().detach().cpu().item()

    return scores
