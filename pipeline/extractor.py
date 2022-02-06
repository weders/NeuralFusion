import torch

from .utils import create_canonical_view

class FusionExtractor(torch.nn.Module):
    '''Module to extract current state of feature volume
       into a canonical view defined by the viewing position
       of the new measurement'''

    def __init__(self, config):

        super(FusionExtractor, self).__init__()

        self.config = config
        self.empty_value = 0.0 * torch.ones(config.n_features)

        # prepare extraction for trilinear interpolation
        if config.FUSION.extraction == 'trilinear':
            indices = []
            for i in range(0, 2):
                for j in range(0, 2):
                    for k in range(0, 2):
                        indices.append(torch.Tensor([i, j, k]).view(1, 1, 3))
            indices = torch.cat(indices, dim=0)
            indices = indices.view(1, 8, 3)
            self.index_shift = indices.int()
            self.index_shift = self.index_shift.unsqueeze_(1)

        # init padding
        self.padding = torch.nn.ReplicationPad3d(1)

    def set_empty_value(self, value):
        self.empty_value = value.detach()

    def forward(self, vpoints, veye, depth, grid):

        b, h, w = depth.shape

        # initialize canonical view
        canonical_view, directions = create_canonical_view(vpoints, veye, self.config.FUSION.n_samples)

        # normalize points (for including points in fusion)
        points_normalized = canonical_view - (canonical_view.floor() + 0.5)

        # reshape grid points
        b, n_points, n_samples, dim = canonical_view.shape
        canonical_view = canonical_view.contiguous().view(b, n_samples * n_points, dim)

        # extracting canonical view (extract current state of feature volume)
        if self.config.FUSION.extraction == 'nearest':
            values = self._nearest_neighbour(canonical_view, grid)
        elif self.config.FUSION.extraction == 'trilinear':
            values = self._interpolate(canonical_view, grid)

        # rehspae extract canonical view
        values = values.view(b, n_points * n_samples, self.config.n_features)
        values = values.view(b, n_points, n_samples, self.config.n_features)
        values = values.view(b, h, w, n_samples, self.config.n_features)
        values = values.view(b, h, w, n_samples * self.config.n_features)

        values = values.permute(0, -1, 1, 2)

        output = {'values': values,
                  'points': points_normalized,
                  'directions': directions}

        return output

    def _reshape(self, data):

        b = self.config.batch_size
        h, w = self.config.resy, self.config.resx
        n_features = self.config.n_features

        data = data.permute(1, 0)
        data = data.view(b, n_features, h, w)
        return data

    def _interpolate(self, grid_points, grid):

        if grid.dim() == 4:
            xs, ys, zs, n_features = grid.shape
        if grid.dim() == 5:
            _, xs, ys, zs, n_features = grid.shape

        grid_points = grid_points.squeeze_(0)

        features = torch.ones((grid_points.shape[0], n_features))
        features = features.to(self.device)
        features = self.empty_value * features

        valid = ((grid_points[:, 0] >= 0) &
                 (grid_points[:, 0] < xs) &
                 (grid_points[:, 1] >= 0) &
                 (grid_points[:, 1] < ys) &
                 (grid_points[:, 2] >= 0) &
                 (grid_points[:, 2] < zs))

        valid_idx = valid.nonzero()
        valid_x = valid_idx[:, 0]

        valid_grid_points = grid_points[valid_x, :]

        # padding of indices
        valid_grid_points = valid_grid_points + torch.ones_like(valid_grid_points)
        grid_points_floor = torch.floor(valid_grid_points)

        if grid.dim() == 4:
            grid_padded = grid.permute(-1, 0, 1, 2).unsqueeze_(0)
            grid_padded = self.padding(grid_padded)
            grid_padded = grid_padded.squeeze_(0).permute(1, 2, 3, 0)

        if grid.dim() == 5:
            grid_padded = grid.permute(0, -1, 1, 2, 3).unsqueeze_(0)
            grid_padded = self.padding(grid_padded.squeeze_(0)).unsqueeze_(0)
            grid_padded = grid_padded.squeeze_(0).permute(0, 2, 3, 4, 1)

        # TODO: fix trilinear interpolation

        interpolation_points_d = valid_grid_points - grid_points_floor

        grid_points_floor = grid_points_floor.unsqueeze_(1)

        interpolation_points = grid_points_floor.repeat((1, 8, 1))

        interpolation_points = interpolation_points + self.index_shift
        interpolation_points = interpolation_points.squeeze_(0)
        interpolation_points = interpolation_points.long()

        df = interpolation_points_d

        # init interpolation weights
        weights = torch.sum(torch.zeros_like(interpolation_points), dim=-1)

        # compute weights
        weights[:, 0] = (1 - df[:, 0]) * (1 - df[:, 1]) * (1 - df[:, 2])
        weights[:, 1] = (1 - df[:, 0]) * (1 - df[:, 1]) * df[:, 2]
        weights[:, 2] = (1 - df[:, 0]) * df[:, 1] * (1 - df[:, 2])
        weights[:, 3] = (1 - df[:, 0]) * df[:, 1] * df[:, 2]
        weights[:, 4] = df[:, 0] * (1 - df[:, 1]) * (1 - df[:, 2])
        weights[:, 5] = df[:, 0] * (1 - df[:, 1]) * df[:, 2]
        weights[:, 6] = df[:, 0] * df[:, 1] * (1 - df[:, 2])
        weights[:, 7] = df[:, 0] * df[:, 1] * df[:, 2]

        # reshape indices
        n, i, d = interpolation_points.shape
        interpolation_points = interpolation_points.view(n * i, d)

        # extract features
        features_to_interpolate = grid_padded[interpolation_points[:, 0],
                                              interpolation_points[:, 1],
                                              interpolation_points[:, 2], :]
        features_to_interpolate = features_to_interpolate.view(n, i, n_features)

        # reshape weights
        weights = weights.unsqueeze_(-1)

        # interpolate features
        valid_features = torch.sum(weights * features_to_interpolate, dim=1)

        features[valid_x, :] = valid_features

        return features

    def _nearest_neighbour(self, points, grid):

        xs, ys, zs, n_features = grid.shape

        points = points.squeeze_(0)

        values = torch.zeros((points.shape[0], n_features))

        if points.get_device() >= 0:
            values = values.to(points.get_device())
            grid = grid.to(points.get_device())
        else:
            self.empty_value = self.empty_value.cpu()


        points_floor = torch.round(points)
        points_floor = points_floor.long()

        valid = ((points_floor[:, 0] >= 0) &
                 (points_floor[:, 0] < xs) &
                 (points_floor[:, 1] >= 0) &
                 (points_floor[:, 1] < ys) &
                 (points_floor[:, 2] >= 0) &
                 (points_floor[:, 2] < zs))

        valid_idx = valid.nonzero()
        valid_x = valid_idx[:, 0]

        points_floor = points_floor[valid_x, :]

        nearest_neighbour_values = grid[points_floor[:, 0],
                                        points_floor[:, 1],
                                        points_floor[:, 2], :]


        values[valid_x, :] = nearest_neighbour_values

        del points_floor, valid_idx, valid_x, nearest_neighbour_values

        return values

    def _unproject(self, depth, extrinsics, intrinsics, origin, resolution):

        b, h, w = depth.shape
        n_points = h * w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float)])

        # flatten grid coordinates and bring them to batch size
        xx = xx.contiguous().view(1, h * w, 1).repeat((b, 1, 1)).to(self.device)
        yy = yy.contiguous().view(1, h * w, 1).repeat((b, 1, 1)).to(self.device)
        zz = depth.contiguous().view(b, h * w, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone()

        # invert intrinsics matrix
        intrinsics_inv = intrinsics.inverse()

        homogenuous = torch.ones((b, 1, n_points))
        homogenuous = homogenuous.to(self.device)

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:, :3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        points_g = (points_w - origin) / resolution

        return points_g



