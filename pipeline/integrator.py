import torch

from datetime import datetime
from torch.nn.functional import normalize


class FusionIntegrator(torch.nn.Module):

    def __init__(self, config):

        super(FusionIntegrator, self).__init__()

        self.config = config

        self.mask = None

        # trilniear interpolation setup
        if config.FUSION.integration == 'trilinear':
            indices = []
            for i in range(0, 2):
                for j in range(0, 2):
                    for k in range(0, 2):
                        indices.append(torch.Tensor([i, j, k]).view(1, 1, 3))
            indices = torch.cat(indices, dim=0)
            indices = indices.view(1, 8, 3)
            self.index_shift = indices.int()
            self.index_shift = self.index_shift.unsqueeze_(1)


    def forward(self, updates, vpoints, veye, mask, grid, count):

        n_features = self.config.n_features - int(self.config.use_count)
        directions = vpoints - veye
        directions = normalize(directions, p=2, dim=2)

        points = [vpoints.float()]

        for i in range(1, self.config.FUSION.n_samples + 1):
            point = vpoints + i * 1. * directions
            pointN = vpoints - i * 1. * directions

            points.append(point)
            points.insert(0, pointN)

        points = torch.stack(points, dim=1)
        grid_points = points

        b, n_samples, n_points, dim = grid_points.shape
        grid_points = grid_points.view(b, n_samples * n_points, dim)

        interpolation_start = datetime.now()
        if self.config.FUSION.integration == 'trilinear':
            p, w = self._interpolate(grid_points)
        elif self.config.FUSION.integration == 'nearest':
            p, w = self._nearest_neighbour(grid_points)
        interpolation_end = datetime.now()
        #print('     Integration: Interpolation Time:', interpolation_end - interpolation_start)

        # reshaping updates
        u = updates.permute(0, 2, 3, 1).view(b, n_points, n_samples, n_features)
        u = u.contiguous().view(n_points * n_samples, n_features)

        # reshaping mask
        mask = torch.stack(n_samples * [mask])
        mask = mask.view(grid_points.shape[0])
        if self.config.FUSION.integration == 'trilinear':
            u = u.unsqueeze_(1).repeat(1, 8, 1)
            mask = mask.unsqueeze_(-1).unsqueeze_(-1).repeat(1, 8, 1)
        elif self.config.FUSION.integration == 'nearest':
            mask = mask.unsqueeze_(-1).unsqueeze_(-1)
            u = u.unsqueeze_(1)

        # reshaping for interpolation and aggregation
        assert u.shape[0] == w.shape[0]
        assert u.shape[0] == p.shape[0]
        assert u.shape[1] == w.shape[1]
        assert u.shape[1] == p.shape[1]

        # reshaping
        n, i, d = u.shape
        u = u.view(n * i, d)
        n, i, d = w.shape
        w = w.view(n * i, d)
        n, i, d = p.shape
        p = p.view(n * i, d)

        n, i, d = mask.shape
        mask = mask.view(n * i, d).squeeze_(-1)

        # filter invalid points
        xs, ys, zs, nfg = grid.shape

        valid = ((p[:, 0] >= 0) &
                 (p[:, 0] < xs) &
                 (p[:, 1] >= 0) &
                 (p[:, 1] < ys) &
                 (p[:, 2] >= 0) &
                 (p[:, 2] < zs))

        valid = valid & mask.int().bool()

        valid_idx = valid.nonzero()
        valid_x = valid_idx[:, 0]

        u = u[valid_x, :]
        w = w[valid_x, :]
        p = p[valid_x, :]

        p = p.long()

        aggregation_start = datetime.now()
        aggregation_values, aggregation_indices = self._aggregate(grid, u, p, w)
        aggregation_values = normalize(aggregation_values, dim=-1, p=2)
        aggregation_end = datetime.now()
        #print('     Integration: Aggregation Time:', aggregation_end - aggregation_start)

        integration_start = datetime.now()
        grid, count = self._integrate(grid, aggregation_values, aggregation_indices, count)
        integration_end = datetime.now()
        #print('     Integration: Integration Time:', integration_end - integration_start)

        del u, w, p,
        return grid, aggregation_indices, count

    def reset(self):
        self.mask = None

    def get_mask(self):
        return self.mask

    def _interpolate(self, grid_points):

        grid_points = grid_points.squeeze_(0)

        valid_grid_points = grid_points

        grid_points_floor = torch.floor(valid_grid_points)
        grid_points_floor = grid_points_floor.unsqueeze_(1)

        interpolation_points = grid_points_floor.repeat((1, 8, 1))
        interpolation_points = interpolation_points + self.index_shift
        interpolation_points = interpolation_points.squeeze_(0)

        grid_points_floor = grid_points_floor.squeeze_(1)

        df = valid_grid_points - grid_points_floor

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

        # reshape weights
        weights = weights.unsqueeze_(-1)

        return interpolation_points, weights

    def _nearest_neighbour(self, points):

        points = points.squeeze_(0)

        # was floor, now trying round
        points_floor = torch.round(points).unsqueeze_(1)
        weights = torch.ones_like(points_floor[:, :, 0]).unsqueeze_(-1)

        return points_floor, weights

    def _upsample(self, update):

        intermediate = torch.zeros(update.shape[0], self.config.n_features, 1, 1, 1).to(self.device)
        intermediate[:, :, 0, 0, 0] = update
        update = self.integration_kernel.forward(intermediate)

        return update

    def _aggregate(self, grid, updates, points, weights=None):

        xs, ys, zs, nfg = grid.shape
        n_features = nfg - self.config.use_count

        aggregator_features = torch.sparse_coo_tensor(points.T, updates, size=[xs, ys, zs, n_features])
        aggregator_count = torch.sparse_coo_tensor(points.T, torch.ones_like(updates[:, 0]).unsqueeze_(-1), size=[xs, ys, zs, 1])

        # merge features and counts
        aggregator_features = aggregator_features.coalesce()
        aggregator_count = aggregator_count.coalesce()

        aggregator_indices = aggregator_count.indices().T
        aggregator_count = aggregator_count.values()
        aggregator_features = aggregator_features.values()

        # average features
        aggregator_features = aggregator_features / aggregator_count

        return aggregator_features, aggregator_indices

    def _integrate(self, volume, updates, points, count):

        if self.config.use_count:
            weights = volume[points[:, 0], points[:, 1], points[:, 2], -1].clone()
            weights = weights.unsqueeze_(-1)

            # compute updates
            updates = (weights * volume[points[:, 0], points[:, 1], points[:, 2], :-1] + updates)
            updates = updates / (weights + 1.)
            
            # assign new values
            volume[points[:, 0], points[:, 1], points[:, 2], :-1] = updates
            volume[points[:, 0], points[:, 1], points[:, 2], -1] += 1
        else:
            weights = count[points[:, 0], points[:, 1], points[:, 2]]
            weights = weights.unsqueeze_(-1)

            if updates.get_device() >= 0:
                weights = weights.to(updates.get_device())

            volume[points[:, 0], points[:, 1], points[:, 2], :] = (weights * volume[points[:, 0], points[:, 1], points[:, 2], :] + updates) / (weights + 1)
            count[points[:, 0], points[:, 1], points[:, 2]] += 1

        return volume, count

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

        homogenuous = torch.ones((b, 1, n_points)).to(self.device)

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        points_g = (points_w - origin) / resolution

        return points_g



