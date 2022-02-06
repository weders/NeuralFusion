import torch

import numpy as np

# from data3d.datasets.tankstemples import TanksTemples

from dataset import ShapeNet, ModelNet, TanksTemples
from dataset.utils.utils import ToTensor

from torch.utils.data import DataLoader

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure

from utils.geometry import interpolate

from easydict import EasyDict

def get_loss_mask(latent_grid, iterations, mode='all', old_state=None, return_orig=False):

    if mode == 'updated':
        assert old_state is not None
        diff = latent_grid - old_state
        diff = torch.abs(diff)
        summed_features = torch.sum(diff, dim=1)
    else:
        assert old_state is not None

        diff = latent_grid - old_state
        diff = torch.abs(diff)
        summed_features = torch.sum(diff, dim=1)
        
        summed_features_all = torch.sum(torch.abs(latent_grid), dim=1)
        mask_all = (summed_features_all > 0) & (summed_features == 0)
        mask_all = torch.squeeze(mask_all)

    # compute mask
    mask = torch.where(summed_features > 0,
                       torch.ones_like(summed_features),
                       torch.zeros_like(summed_features))
    mask = mask.squeeze_(0)
    mask = mask.cpu().detach().numpy()

    if mode == 'all':
        n_samples = 100000

        indices = torch.nonzero(mask_all)
        perm = torch.randperm(indices.shape[0])
        perm = perm[:n_samples]
        indices = indices[perm, :]
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    if return_orig:
        orig_mask = np.copy(mask)

    if iterations > 0:
        # get structure for mask dilation
        structure = generate_binary_structure(3, 3)

        # dilation
        mask = binary_dilation(mask, structure=structure, iterations=iterations)

    mask = torch.Tensor(mask)

    if return_orig:
        return mask, torch.Tensor(orig_mask)
    else:
        return mask


def get_translation_mask(latent_grid, config, groundtruth_grid):

    mask = torch.sum(torch.abs(latent_grid), dim=1)[0]

    if config.DATA.key == 'sparse_depth':
        mask = get_loss_mask(latent_grid, config.LOSS.mask_dilation_iterations)

    mask_indices = torch.nonzero(mask)

    factor = int(config.DATABASE.evaluation_resolution / config.DATABASE.feature_grid_resolution)

    xoffset, yoffset, zoffset = torch.meshgrid([torch.arange(0, factor),
                                                torch.arange(0, factor),
                                                torch.arange(0, factor)])
    xoffset = xoffset.contiguous().view(factor ** 3)
    yoffset = yoffset.contiguous().view(factor ** 3)
    zoffset = zoffset.contiguous().view(factor ** 3)

    offset = torch.stack([xoffset, yoffset, zoffset], dim=1).unsqueeze_(0)

    mask_indices = mask_indices.unsqueeze_(1)

    # sampling to higher resolution (groundtruth resolution)
    mask_indices = factor * mask_indices + offset

    n, i, d = mask_indices.shape
    mask_indices = mask_indices.view(n * i, d)
    mask = torch.zeros_like(groundtruth_grid)

    mask[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]] = 1.
    return mask

def get_dataset(config, mode='train'):

    if mode == 'train':
        
        if config.DATA.dataset == 'shapenet':

            dataset = ShapeNet(config.SETTINGS.data_path,
                               scene_list=config.DATA.train_list,
                               transform=ToTensor(),
                               grid_resolution=config.DATA.grid_resolution_load,
                               noise_scale=config.DATA.noise_scale,
                               outlier_scale=config.DATA.outlier_scale,
                               outlier_fraction=config.DATA.outlier_fraction,
                               load_smooth=False)

        elif config.DATA.dataset == 'modelnet':

            dataset = ModelNet(config.SETTINGS.data_path,
                               scene_list=config.DATA.train_list,
                               transform=ToTensor(),
                               split='train',
                               grid_resolution=config.DATA.grid_resolution_load,
                               noise_scale=config.DATA.noise_scale,
                               outlier_scale=config.DATA.outlier_scale,
                               outlier_fraction=config.DATA.outlier_fraction,
                               sparsity=config.DATA.sparsity)
        
        elif config.DATA.dataset == 'tankstemples':
            
            data_config = EasyDict()
            data_config.root_dir = config.SETTINGS.data_path
            data_config.scene = config.DATA.scene
            data_config.resx = config.DATA.resx
            data_config.resy = config.DATA.resy
            data_config.samples = config.DATA.samples
            data_config.stereo_method = config.DATA.stereo_method

            dataset = TanksTemples(data_config)

    elif mode == 'val':

        if config.DATA.dataset == 'shapenet':

            dataset = ShapeNet(config.SETTINGS.data_path,
                               scene_list=config.DATA.val_list,
                               transform=ToTensor(),
                               grid_resolution=config.DATA.grid_resolution_load,
                               noise_scale=config.DATA.noise_scale,
                               outlier_scale=config.DATA.outlier_scale,
                               outlier_fraction=config.DATA.outlier_fraction,
                               load_smooth=False)

        elif config.DATA.dataset == 'modelnet':

            dataset = ModelNet(config.SETTINGS.data_path,
                               scene_list=config.DATA.val_list,
                               split='train',
                               transform=ToTensor(),
                               grid_resolution=config.DATA.grid_resolution_load,
                               noise_scale=config.DATA.noise_scale,
                               outlier_scale=config.DATA.outlier_scale,
                               outlier_fraction=config.DATA.outlier_fraction,
                               sparsity=config.DATA.sparsity)

        elif config.DATA.dataset == 'tankstemples':
            
            data_config = EasyDict()
            data_config.root_dir = config.SETTINGS.data_path
            data_config.scene = config.DATA.scene
            data_config.resx = config.DATA.resx
            data_config.resy = config.DATA.resy
            data_config.samples = config.DATA.samples
            data_config.stereo_method = config.DATA.stereo_method

            dataset = TanksTemples(data_config)

    elif mode == 'test':
        
        if config.DATA.dataset == 'shapenet':

             dataset = ShapeNet(config.SETTINGS.data_path,
                               scene_list=config.DATA.test_list,
                               transform=ToTensor(),
                               grid_resolution=config.DATA.grid_resolution_load,
                               noise_scale=config.DATA.noise_scale,
                               outlier_scale=config.DATA.outlier_scale,
                               outlier_fraction=config.DATA.outlier_fraction,
                               load_smooth=False)
      
        elif config.DATA.dataset == 'scene3d':
            dataset = Scene3D(config.SETTINGS.data_path,
                              scene=config.DATA.scene)

        elif config.DATA.dataset == 'modelnet':
            dataset = ModelNet(config.SETTINGS.data_path,
                               scene_list=config.DATA.test_list,
                               split='test',
                               transform=ToTensor(),
                               grid_resolution=config.DATA.grid_resolution_load,
                               noise_scale=config.DATA.noise_scale,
                               outlier_scale=config.DATA.outlier_scale,
                               outlier_fraction=config.DATA.outlier_fraction,
                               sparsity=config.DATA.sparsity)

        elif config.DATA.dataset == 'tankstemples':
            
            data_config = EasyDict()
            data_config.root_dir = config.SETTINGS.data_path
            data_config.scene = config.DATA.scene
            data_config.resx = config.DATA.resx
            data_config.resy = config.DATA.resy
            data_config.samples = config.DATA.samples
            data_config.stereo_method = config.DATA.stereo_method

            dataset = TanksTemples(data_config)  

    return dataset


def get_data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)


def filter_points(points, shape):
    #print(points.shape)
    # avoiding cuda errors
    valid_points = (points[:, 0] >= 1) & \
                   (points[:, 0] <= shape[0] - 2) & \
                   (points[:, 1] >= 1) & \
                   (points[:, 1] <= shape[1] - 2) & \
                   (points[:, 2] >= 1) & \
                   (points[:, 2] <= shape[2] - 2)
    valid_points = torch.nonzero(valid_points)[:, 0]
    points = points[valid_points, :]
    #print(points.shape)
    return points

def get_training_points(config, mask, groundtruth):

    factor = int(config.DATABASE.grid_resolution / config.DATABASE.feature_grid_resolution)
    factor = 1

    indices = torch.nonzero(mask).unsqueeze_(1)

    n_points = indices.shape[0]

    if config.PIPELINE.RENDERER.superresolve:
        if config.TRAINING.oversampling:
            offsets = torch.rand((n_points, config.TRAINING.n_points, 3))

            points_to_render = indices + offsets
            points_to_render = points_to_render.contiguous().view(
                n_points * config.TRAINING.n_points, 3)
        else:
            xx, yy, zz = torch.meshgrid(torch.arange(factor),
                                        torch.arange(factor),
                                        torch.arange(factor))

            xx = xx.contiguous().view(factor ** 3)
            yy = yy.contiguous().view(factor ** 3)
            zz = zz.contiguous().view(factor ** 3)

            offsets = torch.stack([xx, yy, zz]).permute(-1, 0).unsqueeze_(0)
            offsets = offsets / 4.

            points_to_render = indices + offsets
            points_to_render = points_to_render.contiguous().view(n_points * (factor ** 3), 3)

    else:
        points_to_render = indices.float()
        points_to_render = points_to_render.contiguous().view(
            n_points, 3)

    shape = (mask.shape[-3], mask.shape[-2], mask.shape[-1])
    points_to_render = filter_points(points_to_render, shape)
    #
    # points_to_render = points_to_render[valid_points, :]

    # compute ground-truth indices
    groundtruth_indices = factor * points_to_render

    if config.TRAINING.interpolation:
        groundtruth_indices, groundtruth_weights = interpolate(groundtruth_indices)
        groundtruth_indices = groundtruth_indices.long()

        n, i, d = groundtruth_indices.shape
        groundtruth_indices = groundtruth_indices.view(n * i, d)

        # extract ground-truth values
        groundtruth_values = groundtruth[:,
                                         groundtruth_indices[:, 0],
                                         groundtruth_indices[:, 1],
                                         groundtruth_indices[:, 2]]

        # interpolation
        groundtruth_values = groundtruth_values.view(n, i, 1)
        groundtruth_values = groundtruth_values * groundtruth_weights
        groundtruth_values = groundtruth_values.sum(dim=1)
        groundtruth_values = groundtruth_values.squeeze_(-1)

    else:
        groundtruth_indices = groundtruth_indices.long()

        # extract ground-truth values
        groundtruth_values = groundtruth[:,
                             groundtruth_indices[:, 0],
                             groundtruth_indices[:, 1],
                             groundtruth_indices[:, 2]]
        groundtruth_values = groundtruth_values.squeeze_(-1)

    return points_to_render, groundtruth_values


def get_translation(pipeline, latent_grid, config):

    if not config.PIPELINE.minimal_gpu:
        device = latent_grid.get_device()

    # empty cuda cache
    torch.cuda.empty_cache()

    if latent_grid.shape[-1] < 128:
        index_rendering = False
        x, y, z = torch.meshgrid([torch.arange(0, config.DATA.evaluation_resolution),
                                  torch.arange(0, config.DATA.evaluation_resolution),
                                  torch.arange(0, config.DATA.evaluation_resolution)])

        x = x.contiguous().view(config.DATA.evaluation_resolution ** 3)
        y = y.contiguous().view(config.DATA.evaluation_resolution ** 3)
        z = z.contiguous().view(config.DATA.evaluation_resolution ** 3)

        n_points = config.DATA.evaluation_resolution ** 3

        points = torch.stack([x, y, z], dim=1).float() / config.DATA.factor


    else:
        index_rendering = True
        mask = torch.sum(torch.abs(latent_grid), dim=1)[0]
        indices = torch.nonzero(mask)

        n_points = indices.shape[0]

        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]

        x = x.clamp(0, latent_grid.shape[-3] - 1)
        y = y.clamp(0, latent_grid.shape[-2] - 1)
        z = z.clamp(0, latent_grid.shape[-1] - 1)


        if config.DATA.dataset == 'scene3d':

            valid = (x >= 3) & \
                    (x <= latent_grid.shape[-3] - 4) & \
                    (y >= 3) & \
                    (y <= latent_grid.shape[-2] - 4) & \
                    (z >= 3) & \
                    (z <= latent_grid.shape[-1] - 4)


            valid_indices = torch.nonzero(valid)[:, 0]


            x = x[valid_indices]
            y = y[valid_indices]
            z = z[valid_indices]

            x = x.clamp(3, latent_grid.shape[-3] - 4)
            y = y.clamp(3, latent_grid.shape[-2] - 4)
            z = z.clamp(3, latent_grid.shape[-1] - 4)

            indices = indices[valid_indices, :]

        points = torch.stack([x, y, z], dim=1).float()
        
    if latent_grid.shape[-1] > 128 or config.PIPELINE.minimal_gpu or config.PIPELINE.renderer == 'transformer':
        n_chunks = int(n_points / 2000)
        points = points.chunk(n_chunks, 0)
        rendered_grid = []
        for p in points:
            torch.cuda.empty_cache()
            if not config.PIPELINE.minimal_gpu:
                p = p.to(device)
            with torch.no_grad():
                padding = False if config.DATA.dataset == 'scene3d' else True
                r = pipeline.translate(p, latent_grid, padding).cpu()
                del p
            rendered_grid.append(r.clone())
        rendered_grid = torch.cat(rendered_grid, dim=0)

    else:
        points = points.to(device)
        torch.cuda.empty_cache()
        rendered_grid = pipeline.translate(points, latent_grid)

    rendered_grid = rendered_grid.squeeze_(-1).squeeze_(-1).squeeze_(-1)

    # if more than one output channel, permute channels
    if len(rendered_grid.shape) > 1:
        rendered_grid = rendered_grid.permute(1, 0)

    if not index_rendering:
        if config.PIPELINE.RENDERER.occ_head:
            rendered_grid = rendered_grid.view(1, 2,
                                               config.DATA.evaluation_resolution,
                                               config.DATA.evaluation_resolution,
                                               config.DATA.evaluation_resolution)
        else:
            rendered_grid = rendered_grid.view(1, 1,
                                               config.DATA.evaluation_resolution,
                                               config.DATA.evaluation_resolution,
                                               config.DATA.evaluation_resolution)

        del points, x, y, z

    else:
        container = torch.zeros(1, 2, latent_grid.shape[-3], latent_grid.shape[-2], latent_grid.shape[-1])

        if rendered_grid.get_device() >= 0:
            container = container.to(rendered_grid.get_device())

        container[0, 0, indices[:, 0], indices[:, 1], indices[:, 2]] = rendered_grid[0, :]
        container[0, 1, indices[:, 0], indices[:, 1], indices[:, 2]] = rendered_grid[1, :]
        container[0, -1, indices[:, 0], indices[:, 1], indices[:, 2]] = rendered_grid[-1, :]


        rendered_grid = container
    return rendered_grid