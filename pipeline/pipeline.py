import torch

from .extractor import FusionExtractor
from .model import FusionNet
from .integrator import FusionIntegrator
from .extractor import FusionExtractor
from .translator import Translator
from .utils import unproject


class FusionPipeline(torch.nn.Module):

    def __init__(self, config):

        super(FusionPipeline, self).__init__()

        self.config = config

        self._extractor = FusionExtractor(config)
        self._fusion = eval(config.FUSION.net)(config)
        self._integrator = FusionIntegrator(config)
        self._translator = Translator(config)

    def _unproject(self, depth, extrinsics, intrinsics, origin, resolution):
        if not self.config.minimal_gpu:
            return unproject(depth, extrinsics, intrinsics, origin, resolution, depth.get_device())
        else:
            return unproject(depth, extrinsics, intrinsics, origin,
                             resolution, torch.device("cpu"))

    def reset(self):
        self._integrator.reset()

    def get_mask(self):
        return self._integrator.get_mask()

    def forward(self, data):

        depth = data['input']
        depth[data['original_mask'] == 0.] = 0.

        points = self._unproject(depth,
                                 data['extrinsics'],
                                 data['intrinsics'],
                                 data['origin'],
                                 resolution=data['resolution'])

        # return points

        eye = data['extrinsics'][:, :3, 3]
        eye = (eye - data['origin']) / data['resolution']


        extraction = self._extractor.forward(points,
                                                eye,
                                                depth,
                                                data['volume'])

     
        n_samples = 2 * self.config.FUSION.n_samples + 1
        n_features = self.config.n_features

        extraction['values'] = extraction['values'].contiguous().view(1, 240, 320, n_features, n_samples)
        extraction['points'] = extraction['points'].contiguous().view(1, 240, 320, 3, n_samples)
        extraction['directions'] = extraction['directions'].contiguous().view(1, 240, 320, 3)

        if self.config.FUSION.uncertainty:
            fusion_input = self._prepare_fusion_input(extraction, depth, data['confidence'])
        else:
            fusion_input = self._prepare_fusion_input(extraction, depth)

        if self.config.minimal_gpu:
            fusion_input = fusion_input.to(self.config.device)

        updates = self._fusion.forward(fusion_input)

        if self.config.minimal_gpu:
            updates = updates.cpu()

        updated_grid, points, count = self._integrator(updates,
                                                        points,
                                                        eye,
                                                        data['original_mask'],
                                                        data['volume'],
                                                        data['count'])

        updated_grid = updated_grid.unsqueeze_(0).permute(0, 4, 1, 2, 3)

        return updated_grid, updates, count

    def translate(self, points, grid, padding=True):
        return self._translator.forward(points, grid, padding)

    def _prepare_fusion_input(self, extractions, depth, uncertainty=None):

        b, h, w, n_features, n_samples = extractions['values'].shape

        input = []

        for i in range(n_samples):
            if self.config.FUSION.position:
                input.append(extractions['points'][:, :, :, :, i])
            if self.config.FUSION.direction:
                input.append(extractions['directions'][:, :, :, :])
            input.append(extractions['values'][:, :, :, :, i])

        input = torch.cat(input, dim=3)
        input = input.permute(0, -1, 1, 2)

        if self.config.FUSION.depth:
            depth = depth.unsqueeze_(0)
            input = torch.cat([depth, input], dim=1)

        if self.config.FUSION.uncertainty:
            confidence = torch.exp(-1. * uncertainty).unsqueeze_(0)
            input = torch.cat([confidence, input], dim=1)

        return input