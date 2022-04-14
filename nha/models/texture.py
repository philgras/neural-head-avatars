from nha.util.general import count_tensor_entries

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiTexture(nn.Module):
    """
    stores several uv texture maps and handles sampling from them based on uv idx
    """

    def __init__(self, *texture_maps):
        """
        text
        :param texture_maps: list of texture tensors of shape C x H_i x W_i
        """

        for t in texture_maps:
            assert len(t.shape) == 3
            assert t.shape[0] == texture_maps[0].shape[0]  # all textures must have same channel count

        super().__init__()
        self.maps = nn.ParameterList([nn.Parameter(t.unsqueeze(0), requires_grad=True) for t in texture_maps])
        self.C = texture_maps[0].shape[0]

    def forward(self, uv_coords, uv_idcs):
        """
        uv_coords of shape
        :param uv_coords: N x H x W x 2 normalized to -1, ... +1
        :param uv_idcs: N x H x W
        :return: N x C x H x W
        """

        assert len(uv_coords.shape) == 4
        assert len(uv_idcs.shape) == 3
        assert uv_coords.shape[:-1] == uv_idcs.shape

        N, H, W, _2 = uv_coords.shape
        ret = torch.zeros(N, H, W, self.C, device=uv_coords.device, dtype=uv_coords.dtype)
        for i, map in enumerate(self.maps):
            mask = (uv_idcs == i)
            ret[mask] = F.grid_sample(map, uv_coords[mask].view(1, 1, -1, 2),
                                      padding_mode="border", align_corners=True)[0, :, 0, :].permute(1, 0)

        return ret.permute(0, 3, 1, 2)

