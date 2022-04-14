import torch
from .siren_mlps import SIRENMLP_DynHead, SirenNormalEncoder as NormalEncoder

class TextureMLP(torch.nn.Module):
    def __init__(self, d_input=3, hidden_features=256, hidden_layers=8, d_dynamic=100,
                 d_dynamic_head=32, n_dynamic_head=1, d_hidden_dynamic_head=128):
        super(TextureMLP, self).__init__()

        self._SIREN = SIRENMLP_DynHead(input_dim=d_input,
                                       output_dim=3,
                                       in_additional_conditioning_channels=d_dynamic,
                                       hidden_layers_feature_size=hidden_features,
                                       hidden_layers=hidden_layers,
                                       n_dynamic=n_dynamic_head, dynamic_feature_size=d_dynamic_head,
                                       d_dynamic=d_hidden_dynamic_head)

    def predict_frequencies_phase_shifts(self, z):
        return self._SIREN.mapping_network(z)

    def forward(self, static_cond, frequencies, phase_shifts, dynamic_conditions=None):
        """
        static conditions: N x C1
        frequencies: N x C2
        phase_shifts: Nx C2
        """
        return torch.tanh(self._SIREN.forward_with_frequencies_phase_shifts(static_cond, frequencies, phase_shifts,
                                                                            dynamic_conditions=dynamic_conditions))







