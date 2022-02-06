import torch

import numpy as np

from graphics.transform import compute_tsdf

def add_noise_to_grid(grid, config):

    # transform to occupancy
    occ = torch.clone(grid)
    occ[grid < 0] = 1
    occ[grid >= 0] = 0

    noise = torch.rand(grid.shape)
    noise[noise < (1. - config.DATA.noise)] = 0.
    noise[noise > 0] = 1.

    occ_noise = occ + noise
    occ_noise = occ_noise.clamp(0, 1)

    grid_noise = occ_noise.detach().numpy()

    dist1 = compute_tsdf(grid_noise.astype(np.float64))
    dist1[dist1 > 0] -= 0.5
    dist2 = compute_tsdf(np.ones(grid_noise.shape) - grid_noise)
    dist2[dist2 > 0] -= 0.5
    grid_noise = np.copy(dist1 - dist2)

    resolution = 1./grid_noise.shape[0]
    grid_noise = resolution * grid_noise

    return torch.tensor(grid_noise)

def get_tsdf(grid):

    dist1 = compute_tsdf(grid.astype(np.float64))
    dist1[dist1 > 0] -= 0.5
    dist2 = compute_tsdf(np.ones(grid.shape) - grid)
    dist2[dist2 > 0] -= 0.5
    tsdf_grid = np.copy(dist1 - dist2)
    resolution = 1. / tsdf_grid.shape[0]
    tsdf_grid = resolution * tsdf_grid

    return tsdf_grid


def get_normal_field(grid):

    [gradx, grady, gradz] = np.gradient(grid)

    # normalize
    norm = np.sqrt(np.power(gradx, 2) + np.power(grady, 2) + np.power(gradz, 2))

    gradx /= (norm + 1.e-08)
    grady /= (norm + 1.e-08)
    gradz /= (norm + 1.e-08)

    gradient = np.stack((gradx, grady, gradz))

    return gradient