import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as ttf
from torchvision.transforms import ToPILImage
import numpy as np
from nha.util.log import get_logger
from typing import *
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur

logger = get_logger(__name__)


class NoSubmoduleWrapper:
    """
    Wrapper for pytorch Module such that module is not assumed to be submodule of parent.
    e.g. consider the following case:

    class ParentModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = TestModule()

    then TestModule which is assumed to also be a nn.Module Subclass is considered as submodule of ParentModule, the
    state dict is inherited, as well as status (eval/train) and device. NoSubmoduleWrapper prevents this from happening.
    Just do:

    class ParentModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = NoSubmoduleWrapper(TestModule())

    and state dicts won't be inherited as well as training states and devices. Keep in mind to set these properties
    manually.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module.__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

    def to(self, device, *args, **kwargs):
        self.module.to(device, *args, **kwargs)


class DecayScheduler:
    def __init__(self, *fixpoints, geometric=False):
        # fixpoints must be pairs of [value, epoch], fixpoints[:, 0] -> values, fixpoints[:, 1] -> epochs
        self._fixpoints = np.array(fixpoints)
        self._fixpoints = self._fixpoints[np.argsort(self._fixpoints[:, 1])]
        self._geometric = geometric

    def get(self, epoch):
        if epoch <= self._fixpoints[0, 1]:
            return self._fixpoints[0, 0]
        elif epoch >= self._fixpoints[-1, 1]:
            return self._fixpoints[-1, 0]
        else:
            i = np.sum(epoch > self._fixpoints[:, 1])
            if self._geometric:

                slope = (self._fixpoints[i, 0] / (self._fixpoints[i - 1, 0] + 1e-14)) ** (
                        1 / (self._fixpoints[i, 1] - self._fixpoints[i - 1, 1]))
                return self._fixpoints[i - 1, 0] * slope ** (epoch - self._fixpoints[i - 1, 1])

            else:
                slope = (self._fixpoints[i, 0] - self._fixpoints[i - 1, 0]) / \
                        (self._fixpoints[i, 1] - self._fixpoints[i - 1, 1])
                return self._fixpoints[i - 1, 0] + slope * (epoch - self._fixpoints[i - 1, 1])


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
        pass

    class C:
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def tensor_distance_img(a, b, norm=2):
    """
    :param a: shape N x C x H x W
    :param b: shape N x C x H x W
    :param norm: which norm to use
    :return: N x 3 x H x W
    """
    assert a.shape == b.shape

    dist = torch.norm(a - b, p=norm, dim=1)
    dist = (dist - dist.min()) / (dist.max() - dist.min())
    cmap = plt.get_cmap("plasma")
    colors = torch.from_numpy(cmap(dist.cpu().numpy()))[..., :3].to(a.device).permute(0, 3, 1, 2)

    return colors


def IoU(x: torch.BoolTensor, y: torch.BoolTensor, eps: float = 1e-7):
    """
    calculates IoU score for boolean mask tensors
    :param x: boolean tensor of shape N x C x H x W
    :param y: boolean tensor of shape N x C x H x W
    :param eps: demoninator offset for numerical stability
    :return: float tensor of shape N, C
    """
    assert x.dtype == torch.bool
    assert y.dtype == torch.bool
    assert x.shape == y.shape

    intersection = x & y
    union = x | y

    iou = intersection.flatten(start_dim=2).sum() / (union.flatten(start_dim=2).sum() + eps)
    return iou


def get_gaussian_kernel1d(kernel_size: int, sigma: float, device="cpu") -> torch.Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def seperated_gaussian_blur(img, kernel_size=None, sigma=1.):
    """
    performs 2d gaussian blurring but instead of applying one 2d kernel applies 2 1d kernels to improve performance
    :param img:
    :param sigma:
    :param kernel_size:
    :return:
    """
    if kernel_size is None:
        kernel_size = closest_odd_int(6 * sigma)
    assert kernel_size % 2 == 1

    raw_kernel = get_gaussian_kernel1d(kernel_size, sigma, device=img.device)  # (kernel_size,)
    kernel_hor = raw_kernel.expand(img.shape[-3], 1, 1, kernel_size)
    kernel_vert = kernel_hor.permute(0, 1, 3, 2)

    out = torch.nn.functional.pad(img, [int((kernel_size - 1) / 2)] * 4, mode="reflect")  #
    out = torch.nn.functional.conv2d(out, weight=kernel_hor, stride=1, padding=0, groups=out.shape[-3])
    out = torch.nn.functional.conv2d(out, weight=kernel_vert, stride=1, padding=0, groups=out.shape[-3])
    return out


def blur_tensors(*tensors: List, sigma=0.):
    tensors = list(tensors)
    if sigma == 0:
        return tensors
    else:
        kernel_size = closest_odd_int(sigma * 6)
        for i in range(len(tensors)):
            tensors[i] = gaussian_blur(tensors[i], kernel_size=kernel_size, sigma=sigma)
        return tensors


def closest_odd_int(x):
    return int((x // 2) * 2 + 1)


def count_tensor_entries(tensor):
    return np.prod([*tensor.shape])


def range2_to_range1(tensor):
    """
    converts an image tensor with range -1 ... +1 to image tensor of range 0 ... 1
    out = in * .5 +.5
    :param tensor:
    :return:
    """
    return tensor * .5 + .5


def range1_to_range2(tensor):
    """
    converts an image tensor with range 0 ... 1 to image tensor of range -1 ... 1
    out = in * 2 - 1
    :param tensor:
    :return:
    """
    return tensor * 2. - 1


def fill_tensor_background(tensor, fg_mask, bg_color=[1., 1., 1.]):
    """
    fills tensor background with specified backround color
    differentiable wrt mask and tensor
    :param tensor:  tensor of shape N x C x H x W
    :param fg_mask: float tensor of shape N x 1 x H x W with range 0 ... 1
    :param bg_color: iterable of length 3
    :return:
    """
    assert fg_mask.dtype == torch.float
    assert len(bg_color) == tensor.shape[1]

    tensor = tensor * fg_mask + torch.tensor(bg_color, device=tensor.device,
                                             dtype=tensor.dtype).view(1, len(bg_color),
                                                                      1,
                                                                      1) * (
                     1. - fg_mask)
    return tensor


def cat_torch_dicts(*dicts, dim=0):
    """
    concatenates dictionary of torch tensors along specified dimension
    :param dicts:
    :param dim:
    :return:
    """

    # handling special cases where first dict is empty and where nr of given dicts == 1
    if len(dicts[0].keys()) == 0:
        return cat_torch_dicts(*dicts[1:], dim=dim)
    if len(dicts) == 1:
        return dicts[0]

    ret = dicts[0].copy()
    for i in range(1, len(dicts)):
        for key, val in dicts[i].items():
            if isinstance(val, dict):
                ret[key] = cat_torch_dicts(ret[key], val)
            elif isinstance(val, list):
                assert dim == 0
                ret[key] = ret[key] + val
            elif isinstance(val, np.ndarray):
                ret[key] = np.concatenate((ret[key], val), axis=dim)
            elif isinstance(val, torch.Tensor):
                ret[key] = torch.cat((ret[key], val), dim=dim)
            else:
                logger.debug(f"Data type {type(val)} of key '{key}' was not understood by cat_torch_dicts(). Used value"
                             f" from last dictionary for resulting dict.")
                ret[key] = val

    return ret


def unsqueeze_torch_dict(d, dim=0, device="cpu"):
    """
    Recursively unsqueezes torch tensors in dict at given dimension. ATTENTION: not quite clear how far this operation
    works inplace or if it makes copies of tensor
    :param d:
    :param dim:
    :param device: specifies on which device to put tensors that are created from scalars
    :return:
    """
    ret = dict()
    for key, val in d.items():
        if isinstance(val, dict):
            ret[key] = unsqueeze_torch_dict(val, dim=dim)
        elif isinstance(val, float) or isinstance(val, int) or isinstance(val, np.int_):
            if dim != 0:
                raise ValueError(
                    f"Unsqueeze dimension {dim} was specified but dictionary contained scalar value.")
            ret[key] = torch.tensor([val], device=device)
        elif isinstance(val, str):
            assert dim == 0
            ret[key] = np.array([val])
        elif isinstance(val, torch.Tensor):
            ret[key] = torch.unsqueeze(val, dim)
        else:
            ret[key] = val
    return ret


def squeeze_torch_dict(d, dim=0):
    ret = dict()
    for key, val in d.items():
        if isinstance(val, dict):
            ret[key] = unsqueeze_torch_dict(val, dim=dim)
        elif isinstance(val, torch.Tensor):
            ret[key] = val.squeeze(dim)
        else:
            ret[key] = val

    return ret


def unstack_dict(d):
    """
    inverse function of stack_dicts()
    from dict with values = tensor of shape N x C -> list of dicts with values=tensor of shape C
    :param d:
    :return:
    """
    N = list(d.values())[0].shape[0]
    ret = [dict() for i in range(N)]

    for key, val in d.items():
        for i in range(N):
            ret[i][key] = val[i]

    return ret


def stack_dicts(*dicts, dim=0):
    """
    stacks dicts along given dimension. To some limitations able to deal with strings, lists, np.arrays and so on
    :param dicts:
    :param dim:
    :return:
    """
    ret = []
    for i in range(len(dicts)):
        ret.append(unsqueeze_torch_dict(dicts[i], dim=dim))
    ret = cat_torch_dicts(*ret, dim=dim)
    return ret


def erode_mask(mask: torch.Tensor, margin: int):
    """
    removes margin from mask with float entries 0. / 1. ATTENTION: Non-discrete values are not supported!
    :param mask: torch tensor of shape N x 1 x H x W with entries 0. / 1.
    :param margin: margin that will be removed from mask,
    :return:
    """
    return 1 - dilate_mask(1 - mask, margin)


def dilate_mask(mask: torch.Tensor, margin: int):
    """
    adds margin to mask with float entries 0. / 1. ATTENTION: Non-discrete values are not supported!
    if margin == -1: returns ones_like(mask)
    :param mask: torch tensor of shape N x 1 x H x W with entries 0. / 1.
    :param margin: margin that will be added to mask,
    :return: 
    """
    assert isinstance(margin, int)
    if margin == 0:
        return mask
    elif margin == -1:
        return torch.ones_like(mask)
    else:
        kernel_size = margin * 2 + 1
        ret = nnF.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=margin,
                             count_include_pad=False,
                             divisor_override=1)
        ret[ret >= 1] = 1
        ret[ret < 1] = 0
        return ret


def save_torch_img(img, path):
    ToPILImage()(img.detach().cpu()).save(path)


def dict_2_device(d, device):
    """
    Recursively sends all tensors in dict to device ATTENTION: not quite clear how far this operation
    works inplace or if it makes copies of tensor
    :param d:
    :param dim:
    :return:
    """
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = dict_2_device(val, device)
        elif isinstance(val, list):
            d[key] = list_2_device(val, device)
        elif isinstance(val, torch.Tensor):
            d[key] = val.to(device)
    return d


def dict_2_dtype(d, dtype):
    """Inplace tensor casting"""
    for key, val in d.items():
        if isinstance(val, torch.Tensor):
            d[key] = val.to(dtype)
    return d


def list_2_device(l, device):
    for i, val in enumerate(l):
        if isinstance(val, torch.Tensor):
            l[i] = val.to(device)
        elif isinstance(val, list):
            l[i] = list_2_device(val, device)
        elif isinstance(val, dict):
            l[i] = dict_2_device(val, device)
    return l


def masked_gaussian_blur(img, mask, kernel_size, sigma, blur_fc=ttf.gaussian_blur, eps=1e-7):
    """
    applies gaussian blurring to img tensor while only using color values on mask. Background fading effects on the mask
    edges are solved by renormalization. Background of blurred image is set to 0
    :param img: N x C x H x W
    :param mask: N x 1 x H x W with entries being either 0. or 1.
    :param sigma:
    :param kernel_size:
    :param blur_fc: optional blurring function with same signature as torchvision.transforms.functional.gaussian_blur():
                    input: (img, kernel_size, sigma) and output: (blurred_img)
    :param eps: denominator stabilisator
    :return:
    """
    assert mask.shape == img[:, :1].shape
    assert mask.shape[1] == 1

    mask = mask.detach()
    img = img * mask
    img_mask = torch.cat((img, mask), dim=1)
    blurred_img_mask = blur_fc(img=img_mask, kernel_size=kernel_size, sigma=sigma)
    img = blurred_img_mask[:, :-1] / (blurred_img_mask[:, -1:] + eps)
    img = img * mask
    return img


@torch.enable_grad()
def softmask_gradient(x, mask):
    """
    masks gradient values on tensor x based on masking values in tensor mask. Can be used to apply gradient updates
    on a specified region of the tensor only

    :param x: tensor of shape N x C x H x W that is part of gradient graph
    :param mask: tensor of shape N x 1 x H x W with values ranging from 0...1 determining the gradient masking weight
    :return: tensor with same content as x but gradients are weighted with with values from mask
    """
    mask = mask.detach()
    y = x * mask + x.detach() * (1 - mask)
    return y


def get_mask_bbox(mask):
    """
    Computes the bounding box around the binary mask
    :param mask: 2D binary mask
    :return: row_min, row_max, col_min, col_max
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.flatnonzero(rows)[[0, -1]]
    xmin, xmax = np.flatnonzero(cols)[[0, -1]]

    return ymin, ymax, xmin, xmax


def write_obj(fp: str, verts: np.ndarray, face_vert_idcs: np.ndarray, face_uv_idcs: np.ndarray, uvs: np.ndarray):
    """
    writes an obj file. More specific information about how obj file format works in pytorch3d/io/obj_io.py:load_obj()

    :param fp: filepath
    :param verts: V x 3
    :param face_vert_idcs: F x 3
    :param face_uv_idcs: F x 3
    :param uvs: T x 2
    :return:
    """
    assert len(verts.shape) == 2
    assert verts.shape[-1] == 3

    lines = []

    # saving vertices
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")

    # saving uvcoordinates
    for vt in uvs:
        lines.append(f"vt {vt[0]} {vt[1]}")

    # saving faces
    for f_v, f_uv in zip(face_vert_idcs, face_uv_idcs):
        f_v += 1  # obj files are 1-indexed
        f_uv += 1
        lines.append(f"f {f_v[0]}/{f_uv[0]} {f_v[1]}/{f_uv[1]} {f_v[2]}/{f_uv[2]}")

    # adding linebreaks
    for i in range(len(lines) - 1):
        lines[i] = lines[i] + "\n"

    with open(fp, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return

