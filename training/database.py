import torch
import h5py
import os

import numpy as np

from graphics.voxelgrid import FeatureGrid, Voxelgrid

class Database(torch.utils.data.Dataset):

    def __init__(self, dataset, config):

        super(Database, self).__init__()

        self.config = config

        self.grids_gt = {}
        self.grids_latent = {}
        self.grids_count = {}

        self._initialize(dataset)

    def _initialize(self, dataset):

        for s in dataset.scenes:

            # grid = dataset.get_tsdf(s, resolution=self.config.DATA.grid_resolution)

            grid = dataset.get_grid(s)

            if np.min(grid.volume) >= 0:
                grid.transform('normal')
                grid.volume *= grid.resolution

            if self.config.grid_resolution_load != self.config.grid_resolution:
                assert self.config.grid_resolution < self.config.grid_resolution_load
                downsampled_grid = Voxelgrid(resolution=1./self.config.grid_resolution)

                data = grid.volume
                downsampled_data = data[::1, ::1, ::1]

                downsampled_grid.from_array(downsampled_data, grid.bbox)
                grid = downsampled_grid

            self.grids_gt[s] = grid
            self.grids_latent[s] = FeatureGrid(resolution=1./self.config.feature_grid_resolution,
                                               bbox=grid.bbox,
                                               origin=grid.origin,
                                               n_features=self.config.n_features)

            self.grids_count[s] = np.sum(np.zeros(self.grids_latent[s].shape), axis=-1)
            self.init_volume = 0.0 * np.ones(self.grids_latent[s].shape)
            # self.init_volume = 0.1 * np.ones(self.grids_latent[s].shape)

    def set_initialization(self, data):
        self.init_volume = data

    def __getitem__(self, item):

        sample = {}
        sample['gt'] = self.grids_gt[item]
        sample['latent'] = self.grids_latent[item]
        sample['count'] = self.grids_count[item]

        return sample

    def update(self, item, grid, count):
        self.grids_latent[item].data = grid
        self.grids_count[item] = count

    def reset(self):
        for k in self.grids_latent.keys():
            self.grids_latent[k].data = self.init_volume
            self.grids_count[k] = np.sum(np.zeros(self.grids_latent[k].shape), axis=-1)

    def save(self, scene_id, path):

        filename = '{}.volume.hf5'.format(scene_id.replace('/', '.'))
        gtname = '{}.weights.hf5'.format(scene_id.replace('/', '.'))

        with h5py.File(os.path.join(path, filename), 'w') as hf:
            hf.create_dataset("TSDF",
                              shape=self.scenes_est[scene_id].data.shape,
                              data=self.scenes_est[scene_id].data)

        with h5py.File(os.path.join(path, gtname), 'w') as hf:
            hf.create_dataset("TSDF",
                              shape=self.grids_gt[scene_id].shape,
                              data=self.grids_gt[scene_id].volume)



    @property
    def ids(self):
        return self.grids_gt.keys()
