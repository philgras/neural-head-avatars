import torch
from .siren_mlps import SIRENMLP


class OffsetMLP(torch.nn.Module):
    def __init__(self,
                 cond_feats=0,
                 hidden_feats=128,
                 hidden_layers=6,
                 in_feats=0):
        super().__init__()
        self._SIREN = SIRENMLP(input_dim=in_feats,
                                              output_dim=3,
                                              condition_dim=cond_feats,
                                              hidden_dim=hidden_feats,
                                              hidden_layers=hidden_layers)

    def forward(self, x, conditions=None):
        res = self._SIREN.forward(vertices=x, additional_conditioning=conditions)
        return torch.tanh(res)
