import torch
from torch.nn.functional import grid_sample as torch_gridsample
from torch import Tensor
from typing import Optional, Any


class ScreenGrad(torch.autograd.Function):
    """
    Calculates screen coordinate gradient for given input image and attaches gradient to given screen coordinate tensor
    ATTENTION: output of this function equals input image and is not depending on given screen coordinate tensor
    Assumes input and screen coords to have same dimensions and assumes x axis to go from left to right, y to go from
    top to bottom. Also assumes screen coords to be normalized to the range -1 ... +1

    ATTENTION: Not complying with above conventions won't necessarily raise an error but may lead to undesired gradients
    """
    @staticmethod
    def forward(ctx,
                input: Tensor,
                screen_coords) -> Tensor:
        """
        :param ctx:
        :param input: N x C x H x W
        :param screen_coords: N x H x W x 2; x,y ranging from -1,-1 to +1, +1
        :return:
        """
        assert input[:, 0].shape == screen_coords.shape[:-1]
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        input, = ctx.saved_tensors
        grad_input = grad_screen_coords = None

        # grad_output has shape N x C x H x W

        N, C, H, W = input.shape

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

        if ctx.needs_input_grad[1]:
            grad_screen_coords = torch.zeros(N, H, W, 2, dtype=grad_output.dtype, device=grad_output.device)

            # Horizontal gradients
            grad_screen_coords[:, :, 1:-1, 0] = ((input[..., 2:] - input[..., :-2]) / 2
                                                 * grad_output[..., 1:-1]).sum(dim=1)
            grad_screen_coords[:, :, [0, -1], 0] = ((input[..., [1, -1]] - input[..., [0, -2]])
                                                    * grad_output[..., [0, -1]]).sum(dim=1)

            # Vertical gradients
            grad_screen_coords[:, 1:-1, :, 1] = ((input[..., 2:, :] - input[..., :-2, :]) / 2
                                                 * grad_output[..., 1:-1, :]).sum(dim=1)
            grad_screen_coords[:, [0, -1], :, 1] = ((input[..., [1, -1], :] - input[..., [0, -2], :])
                                                    * grad_output[..., [0, -1], :]).sum(dim=1)

            # Scaling gradients to NDC grid space space from -1 to +1
            grad_screen_coords[..., 0] *= W / 2.
            grad_screen_coords[..., 1] *= H / 2.

        return grad_input, grad_screen_coords

# DEFINING DIFFERENTIABLE FUNCTION FOR LATER USE
screen_grad = ScreenGrad.apply
