import torch
from torch import nn

from torch.nn.functional import normalize


class EncoderBlock(nn.Module):
    '''Encoder block for the fusion network in NeuralFusion'''

    def __init__(self, c_in, c_out, activation, resolution):

        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation,
                                   nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    '''Decoder block for the fusion network in NeuralFusion'''

    def __init__(self, c_in, c_out, activation, resolution):

        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation,
                                   nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation)

    def forward(self, x):
        return self.block(x)


class FusionNet(nn.Module):

    def __init__(self, config):

        super(FusionNet, self).__init__()

        self.n_padding = config.FUSION['encoder_filter_size'] // 2
        self.n_points = 2 * config.FUSION['n_samples'] + 1
        self.n_features = config['n_features'] - int(config.use_count)

        # compute layer settings
        self.n_channels_input = self.n_points * (self.n_features + int(config.use_count) + 3 * config.FUSION.position + 3 * config.FUSION.direction)
        self.n_channels_output = self.n_points * self.n_features
        self.n_layers = config.FUSION.n_layers
        self.height = config.FUSION.resy
        self.width = config.FUSION.resx
        resolution = (self.height, self.width)
        activation = eval(config.FUSION.activation)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.n_channels_first = self.n_channels_input + config.FUSION.depth

        # add first encoder block
        self.encoder.append(EncoderBlock(self.n_channels_first,
                                         self.n_channels_input,
                                         activation,
                                         resolution))
        # add first decoder block
        self.decoder.append(DecoderBlock((self.n_layers + 1) * self.n_channels_input + config.FUSION.depth,
                                         self.n_layers * self.n_channels_output,
                                         activation,
                                         resolution))

        # adding model layers
        for l in range(1, self.n_layers):
            self.encoder.append(EncoderBlock(self.n_channels_first + l * self.n_channels_input,
                                             self.n_channels_input,
                                             activation,
                                             resolution))

            self.decoder.append(DecoderBlock(((self.n_layers + 1) - l) * self.n_channels_output,
                                             ((self.n_layers + 1) - (l + 1)) * self.n_channels_output,
                                             activation,
                                             resolution))

    def forward(self, x):

        # encoding
        for enc in self.encoder:
            xmid = enc(x)
            x = torch.cat([x, xmid], dim=1)

        # decoding
        for dec in self.decoder:
            x = dec(x)

        # normalization
        x = x.view(1, self.n_features, self.n_points, self.height, self.width)
        x = normalize(x, p=2, dim=1)
        x = x.view(1, self.n_features * self.n_points, self.height, self.width)

        return x