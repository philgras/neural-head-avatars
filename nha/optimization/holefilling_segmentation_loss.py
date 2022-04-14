import torch
from nha.util.general import closest_odd_int, seperated_gaussian_blur, IoU
from nha.util.screen_grad import screen_grad


def calc_holefilling_segmentation_loss(gt_mask,
                                       pred_mask,
                                       screen_coords,
                                       sigma=.5,
                                       return_iou=False,
                                       channel_weights=None,
                                       grad_weight=None,
                                       sample_weights=None):
    """
    calculates union/holefilling segmentation loss. Supports multi-class segmentation

    :param gt_mask: N x C x H x W with binary values 0. / 1.
    :param pred_mask: N x C x H x W with binary values 0. / 1.
    :param screen_coords:
    :param sigma:
    :param return_iou:
    :param weights: weight list of length C on how to weight different classes. default weights are 1
    :return: scalar loss (+ IoU tensor of shape N x C)
    """
    pix_on_gt_silhouette = screen_grad(gt_mask, screen_coords)
    pix_on_gt_silhouette_mask = pix_on_gt_silhouette.bool()  # N x C x H x W

    gt_mask_bool = gt_mask.bool()  # N x C x H_img x W_img
    pred_mask_bool = pred_mask.bool()  # N x C x H_img x W_img
    pred_mask_gt_union = gt_mask_bool | pred_mask_bool  # N x C x H_img x W_img
    seg_holes = gt_mask_bool & (~pred_mask_bool)  # N x C x H_img x W_img
    backpulling_mask = (~pix_on_gt_silhouette_mask) & pred_mask_bool  # N x C x H_r x W_r

    kernel_size = closest_odd_int(6 * sigma)
    blurred_pred_mask_gt_union = seperated_gaussian_blur(pred_mask_gt_union.float(),
                                                         kernel_size=kernel_size,
                                                         sigma=sigma).clip(min=0, max=1)
    blurred_holes = seperated_gaussian_blur(seg_holes.float(), kernel_size=kernel_size,
                                            sigma=sigma).clip(min=0, max=1)

    hole_closing_alphas = screen_grad(blurred_holes, screen_coords)  # N x C x H x W
    backpulling_alphas = screen_grad(blurred_pred_mask_gt_union, screen_coords)  # N x C x H x W

    # weighting
    if channel_weights is None:
        channel_weights = 1.
    else:
        assert len(channel_weights) == gt_mask.shape[1]
        channel_weights = torch.tensor(channel_weights, device=gt_mask.device).view(1, -1, 1, 1)

    if grad_weight is None:
        grad_weight = 1

    if sample_weights is None:
        sample_weights = 1
    else:
        assert len(sample_weights) == gt_mask.shape[0]
        sample_weights = sample_weights.view(-1, 1, 1, 1)

    hole_closing_loss = (1 - hole_closing_alphas) * channel_weights * grad_weight * sample_weights
    backpulling_loss = (1 - backpulling_alphas) * channel_weights * grad_weight * sample_weights

    loss = hole_closing_loss[pred_mask_bool].sum() + backpulling_loss[backpulling_mask].sum()
    loss = loss / (pred_mask.sum() + backpulling_mask.sum())

    if return_iou:
        return loss, IoU(gt_mask_bool, pred_mask_bool)
    else:
        return loss
