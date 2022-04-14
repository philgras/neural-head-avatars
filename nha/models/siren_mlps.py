import torch
from torch import nn
import numpy as np


class SIRENMLP(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=3,
                 hidden_dim=256,
                 hidden_layers=8,
                 condition_dim=100,
                 device=None):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.z_dim = condition_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [FiLMLayer(self.input_dim, self.hidden_dim)] +
            [FiLMLayer(self.hidden_dim, self.hidden_dim) for i in range(self.hidden_layers - 1)]
        )
        self.final_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.mapping_network = MappingNetwork(condition_dim, 256,
                                              len(self.network) * self.hidden_dim * 2)

        self.network.apply(frequency_init(25))
        # self.final_layer.apply(frequency_init(25))
        self.final_layer.weight.data.normal_(0.0, 0.)
        self.final_layer.bias.data.fill_(0.)
        self.network[0].apply(first_layer_film_sine_init)

    def forward_vector(self, input, z):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)

        return sigma

    def forward(self, vertices, additional_conditioning):
        # vertices: in canonical space of flame
        # vertex eval: torch.Size([N, V, 3])
        # map eval:    torch.Size([N, 3, H, W])
        # conditioning
        # torch.Size([N, C])

        # vertex inputs (N, V, 3) -> (N, 3, V, 1)
        vertices = vertices.permute(0, 2, 1)[:, :, :, None]
        b, c, h, w = vertices.shape

        # to vector
        x = vertices.permute(0, 2, 3, 1).reshape(b, -1, c)

        z = additional_conditioning  # .repeat(1, h*w, 1)

        # apply siren network
        o = self.forward_vector(x, z)

        # to image
        o = o.reshape(b, h, w, self.output_dim)
        output = o.permute(0, 3, 1, 2)

        return output[:, :, :, 0].permute(0, 2, 1)


class SIRENMLP_DynHead(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=10,
                 hidden_layers_feature_size=256,
                 hidden_layers=8,
                 in_additional_conditioning_channels=100,
                 use_additional_spatial_embedding=False,
                 device=None,
                 dynamic_feature_size=0, n_dynamic=1, d_dynamic=128):
        super().__init__()
        z_dim = in_additional_conditioning_channels
        hidden_dim = hidden_layers_feature_size

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.d_dynamic = d_dynamic

        self.use_additional_spatial_embedding = use_additional_spatial_embedding
        self.spatial_embeddings_dim = 64  # 96
        self.spatial_features_dim = 32
        if self.use_additional_spatial_embedding:
            self.input_dim += self.spatial_features_dim

        # BODY
        self.network = nn.ModuleList(
            [OwnFiLMLayer(self.input_dim, self.hidden_dim)] +
            [OwnFiLMLayer(self.hidden_dim, self.hidden_dim) for i in range(self.hidden_layers - 1)]
        )

        # STATIC HEAD
        self.final_layer = nn.Linear(self.hidden_dim, self.output_dim)

        # DYNAMIC HEAD
        self.dynamic_head = nn.ModuleList([OwnFiLMLayer(self.hidden_dim + dynamic_feature_size, d_dynamic)] +
                                          [OwnFiLMLayer(d_dynamic, d_dynamic) for i in range(n_dynamic - 1)] +
                                          [nn.Linear(d_dynamic, self.output_dim)])

        self.mapping_network = MappingNetwork(z_dim, 256, (
                len(self.network) * self.hidden_dim + d_dynamic * n_dynamic) * 2, )

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.dynamic_head.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        if self.use_additional_spatial_embedding:
            self.spatial_embeddings = nn.Parameter(
                torch.randn(1, self.spatial_features_dim, self.spatial_embeddings_dim,
                            self.spatial_embeddings_dim) * 0.01)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, dynamic_conditions=None):
        frequencies = frequencies * 15 + 30

        if self.use_additional_spatial_embedding:
            raise NotImplementedError
            # input = (input + 1.0) / 2.0  # [-1,1] -> [0,1]
            shared_features = sample_from_2dgrid(input, self.spatial_embeddings)
            x = torch.cat([shared_features, input], -1)
        else:
            x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)

        if dynamic_conditions is not None:
            x = torch.cat((x, dynamic_conditions), dim=-1)
            for index, layer in enumerate(self.dynamic_head[:-1]):
                start = len(self.network) * self.hidden_dim + index * self.d_dynamic
                end = len(self.network) * self.hidden_dim + (index + 1) * self.d_dynamic
                x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            x = self.dynamic_head[-1](x)
            sigma = sigma + x

        return sigma

    def forward(self):
        raise NotImplementedError


class SirenNormalEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=10, hidden_layers_feature_size=256, hidden_layers=8,
                 device=None):
        super().__init__()
        hidden_dim = hidden_layers_feature_size

        self.device = device
        self.input_dim = input_dim
        self.z_dim = 100
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [FiLMLayer2D(self.input_dim, self.hidden_dim)] +
            [FiLMLayer2D(self.hidden_dim, self.hidden_dim) for i in range(self.hidden_layers - 1)]
        )
        self.final_layer = nn.Conv2d(hidden_dim, self.output_dim, kernel_size=3, padding=1, padding_mode="replicate")

        self.mapping_network = MappingNetwork(self.z_dim, 256, (len(self.network) + 1) * self.hidden_dim * 2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.register_buffer("_static_condition",  # 1 x 100
                             torch.tensor([[-1.10247604, -1.42804892, 0.87488722, 1.15597923, -0.57568966,
                                            -0.04965701, 0.07089813, -2.21527982, 0.13931708, 0.30833226,
                                            1.7272208, 0.40821049, -0.86179693, 0.49263112, 1.1502004,
                                            1.01742589, -0.91245066, -0.43667411, -0.1643309, 1.37222254,
                                            -0.40959126, -1.26752158, 1.84200903, -0.42769857, -1.72281049,
                                            1.54676358, 0.70155848, 1.32231672, -1.55473343, -0.64930216,
                                            1.33731289, 0.68014974, -0.87769585, -1.13550678, -0.24108198,
                                            -0.56245497, -0.26652164, -1.67956008, -0.4283457, 1.26249083,
                                            0.40890499, 0.40677053, -0.44340142, 1.25217541, 2.13999611,
                                            -1.17023326, -1.10413861, -0.30643598, -0.17989996, 0.7573133,
                                            -0.09529626, -1.67288201, -0.56437952, 1.14097756, 0.43888664,
                                            0.73405574, -0.99273702, 0.15076723, 0.65683655, 1.44445144,
                                            -0.32217313, 0.50703575, -0.98423476, -1.13127301, -0.65218117,
                                            0.18305708, 0.26766889, 0.30279705, -2.3601944, -1.4982377,
                                            1.86408925, 1.11687454, -0.2839407, 1.7602057, 0.4117718,
                                            0.34926186, 0.66022188, -0.55205758, 0.03506068, -0.01726678,
                                            -0.18771338, 0.03045248, -0.04338471, 0.01669747, -0.84164925,
                                            0.95326186, 0.52841106, 1.71889631, -1.26329448, -0.19737745,
                                            -1.10392851, -0.39721072, -0.42183207, 1.73825699, -1.24009311,
                                            -0.84449652, 0.43659571, -0.00715666, 0.35093695, 1.20645271]]))

    def forward_vector(self, input, z, additional_condition=None):
        frequencies, phase_shifts = self.mapping_network(z)

        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts,
                                                          additional_condition=additional_condition)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, additional_condition=None):
        frequencies = frequencies * 15 + 30

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            if index == 0 and additional_condition is not None:  # ADDING additional condition
                x[:, :additional_condition.shape[1]] = x[:, :additional_condition.shape[1]] + additional_condition
        sigma = self.final_layer(x)

        return sigma

    def forward(self, normals, z=None, additional_condition=None):
        # normals N x 3 x H x W
        # z torch tensor torch.Size([N, C])

        if z is None:
            z = self._static_condition.expand(normals.shape[0], -1)

        # apply siren network
        o = self.forward_vector(normals, z, additional_condition=additional_condition)

        return o


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift, ignore_conditions=None):
        x = self.layer(x)
        if ignore_conditions is not None:
            cond_freq, cond_phase_shift = freq[:-1], phase_shift[:-1]
            cond_freq = cond_freq.unsqueeze(1).expand_as(x).clone()
            cond_phase_shift = cond_phase_shift.unsqueeze(1).expand_as(x).clone()

            ignore_freq, ignore_phase_shift = freq[-1:], phase_shift[-1:]
            ignore_freq = ignore_freq.unsqueeze(1).expand_as(x)
            ignore_phase_shift = ignore_phase_shift.unsqueeze(1).expand_as(x)

            cond_freq[:, ignore_conditions] = ignore_freq[:, ignore_conditions]
            cond_phase_shift[:, ignore_conditions] = ignore_phase_shift[:, ignore_conditions]
            freq, phase_shift = cond_freq, cond_phase_shift

        else:
            freq = freq.unsqueeze(1).expand_as(x)
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        # print('x', x.shape)
        # print('freq', freq.shape)
        # print('phase_shift', phase_shift.shape)
        # x torch.Size([6, 5023, 256])
        # freq torch.Size([6, 5023, 256])
        # phase_shift torch.Size([6, 5023, 256])
        return torch.sin(freq * x + phase_shift)


class OwnFiLMLayer(nn.Module):
    """
    doesnt expand frequencies and phase shifts but expects them to have same shape as x
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)

        # CHANGED
        # freq = freq.unsqueeze(1).expand_as(x)
        # phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        # print('x', x.shape)
        # print('freq', freq.shape)
        # print('phase_shift', phase_shift.shape)
        # x torch.Size([6, 5023, 256])
        # freq torch.Size([6, 5023, 256])
        # phase_shift torch.Size([6, 5023, 256])
        return torch.sin(freq * x + phase_shift)


class FiLMLayer2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding,
                               padding_mode="replicate")

    def forward(self, x, freq, phase_shift):
        """

        :param x: N x input_dim x H x W
        :param freq: N x hidden_dim
        :param phase_shift: N x hidden_dim
        :return:
        """
        x = self.layer(x)
        freq = freq.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        # print('x', x.shape)
        # print('freq', freq.shape)
        # print('phase_shift', phase_shift.shape)
        # x torch.Size([6, 5023, 256])
        # freq torch.Size([6, 5023, 256])
        # phase_shift torch.Size([6, 5023, 256])
        return torch.sin(freq * x + phase_shift)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
            elif isinstance(m, nn.Conv2d):
                num_input = torch.prod(
                    torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
        elif isinstance(m, nn.Conv2d):
            num_input = torch.prod(
                torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
            m.weight.uniform_(-1 / num_input, 1 / num_input)
