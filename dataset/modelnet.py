import os
import glob
import cv2
import cc3d

import numpy as np

from skimage import io, transform
from torch.utils.data import Dataset
from copy import copy
import random
from graphics import Voxelgrid
import gzip

import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_dilation

from .utils.augmentation import *

import pickle
from .utils.binvox_utils import read_as_3d_array


class ModelNet(Dataset):

    def __init__(self, root_dir,
                 obj=None,
                 model=None,
                 scene_list=None,
                 split='train',
                 n_frames=100,
                 resolution=(240, 320),
                 transform=None,
                 noise_scale=0.05,
                 outlier_scale=5,
                 outlier_fraction=0.99,
                 sparsity=0.1,
                 grid_resolution=128,
                 frequency=1):

        self.root_dir = os.path.expanduser(root_dir)

        self.resolution = resolution
        self.xscale = resolution[0] / 480.
        self.yscale = resolution[1] / 640.

        self.noise_scale = noise_scale
        self.outlier_scale = outlier_scale
        self.outlier_fraction = outlier_fraction

        self.transform = transform

        self.obj = obj
        self.model = model
        self.scene_list = scene_list

        self.sparsity = sparsity

        self.grid_resolution = grid_resolution

        self.n_frames = n_frames

        self.split = split
        self.frequency = frequency

        self._load_frames()

    def _load_frames(self):

        self._scenes = []

        if self.scene_list is None:
            # scene, obj = self.scene.split('/')
            path = os.path.join(self.root_dir, self.obj, self.split,
                                self.model, 'data', '*.depth.png')
            files = glob.glob(path)
            self.frames = []

            if self.n_frames == 100:
                ids = [i for i in range(100)]
            if self.n_frames == 80:
                ids = [i for i in range(100) if i % 5 != 0]
            if self.n_frames == 60:
                ids = [i for i in range(100) if i % 2 == 0] + [1, 13, 25, 37,
                                                               49, 51, 63, 75,
                                                               87, 99]
            if self.n_frames == 40:
                ids = [i for i in range(100) if (i % 2 == 0 and i % 10 != 0)]
            if self.n_frames == 20:
                ids = [i for i in range(100) if i % 5 == 0]

            for i, f in enumerate(files):
                self.frames.append(f.replace('.depth.png', ''))


        else:
            self.frames = []

            with open(self.scene_list, 'r') as file:

                for line in file:
                    try:
                        scene, obj = line.rstrip().split('\t')
                    except:
                        scene, obj = line.rstrip().split(' ')

                    self._scenes.append(os.path.join(scene, obj))

                    path = os.path.join(self.root_dir, scene, self.split, obj,
                                        'data', '*.depth.png')
                    files = glob.glob(path)

                    frames = []
                    for i, f in enumerate(files):
                        frames.append(f.replace('.depth.png', ''))

                    if self.frequency > 1:
                        subsampled = []
                        for i in range(len(frames)):
                            if i % self.frequency == 0:
                                subsampled.append(frames[i])
                        self.frames = self.frames + subsampled
                    else:
                        self.frames = self.frames + frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        frame = self.frames[item]
        pathsplit = frame.split('/')
        sc = pathsplit[-5]
        obj = pathsplit[-3]
        scene_id = '{}/{}'.format(sc, obj)
        sample = {}

        frame_id = frame.split('/')[-1]
        frame_id = int(frame_id)
        sample['frame_id'] = frame_id

        depth = io.imread('{}.depth.png'.format(frame))
        depth = depth.astype(np.float32)
        depth = depth / 1000.
        # depth[depth == np.max(depth)] = 0.

        step_x = depth.shape[0] / self.resolution[0]
        step_y = depth.shape[1] / self.resolution[1]

        index_y = [int(step_y * i) for i in
                   range(0, int(depth.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(depth.shape[0] / step_x))]

        depth = depth[:, index_y]
        depth = depth[index_x, :]

        mask = copy(depth)
        mask[mask == np.max(depth)] = 0
        mask[mask != 0] = 1
        original_mask = copy(mask)
        sample['original_mask'] = copy(mask)
        gradient_mask = binary_dilation(mask, iterations=5)
        mask = binary_dilation(mask, iterations=8)
        sample['mask'] = mask
        sample['gradient_mask'] = gradient_mask

        depth[mask == 0] = 0

        sample['depth'] = depth
        sample['noisy_depth'] = add_kinect_noise(copy(depth),
                                                 sigma_fraction=self.noise_scale)
        sample['noisy_depth_octnetfusion'] = add_depth_noise(copy(depth),
                                                             noise_sigma=self.noise_scale,
                                                             seed=42)
        sample['outlier_depth'] = add_outliers(
            copy(sample['noisy_depth_octnetfusion']),
            scale=self.outlier_scale,
            fraction=self.outlier_fraction)

        sample['sparse_depth'] = add_sparse_depth(copy(depth), percentage=self.sparsity)
        sample['outlier_blob_depth'] = add_outlier_blobs(copy(sample['noisy_depth_octnetfusion']),
                                                         scale=self.outlier_scale,
                                                         fraction=self.outlier_fraction)
        sample['heuristic_noise'] = add_depth_noise(copy(depth),
                                                    noise_sigma=self.noise_scale,
                                                    seed=42)
        sample['heuristic_noise'] = add_gradient_noise(sample['heuristic_noise'], depth)
        sample['heuristic_noise'] = add_outlier_blobs(sample['heuristic_noise'],
                                                      scale=self.outlier_scale,
                                                      fraction=self.outlier_fraction,
                                                      starti=3,
                                                      endi=7)


        intrinsics = np.loadtxt('{}.intrinsics.txt'.format(frame))
        # adapt intrinsics to camera resolution
        scaling = np.eye(3)
        scaling[1, 1] = self.yscale
        scaling[0, 0] = self.xscale

        sample['intrinsics'] = np.dot(scaling, intrinsics)

        extrinsics = np.loadtxt('{}.extrinsics.txt'.format(frame))
        extrinsics = np.linalg.inv(extrinsics)
        sample['extrinsics'] = extrinsics

        sample['scene_id'] = scene_id

        for key in sample.keys():
            if type(sample[key]) is not np.ndarray and type(
                    sample[key]) is not str:
                sample[key] = np.asarray(sample[key])

        if self.transform:
            sample = self.transform(sample)

        return sample
    @property
    def scenes(self):
        return self._scenes

    def get_grid(self, scene):

        sc, obj = scene.split('/')
        if self.grid_resolution == 256:
            filepath = os.path.join(self.root_dir, sc, self.split, obj,
                                    'voxels', '*.binvox')
        else:
            filepath = os.path.join(self.root_dir, sc, self.split, obj,
                                    'voxels', '*.{}.binvox'.format(
                    self.grid_resolution))

        filepath = glob.glob(filepath)[0]

        # filepath = os.path.join(self.root_dir, 'example', 'voxels', 'chair_0256.binvox')

        with open(filepath, 'rb') as file:
            volume = read_as_3d_array(file)

        array = volume.data.astype(np.int)

        # clean occupancy grids from artifacts
        labels_out = cc3d.connected_components(array)  # 26-connected
        N = np.max(labels_out)
        max_label = 0
        max_label_count = 0
        for segid in range(1, N + 1):
            extracted_image = labels_out * (labels_out == segid)
            extracted_image[extracted_image != 0] = 1
            label_count = np.sum(extracted_image)
            if label_count > max_label_count:
                max_label = segid
                max_label_count = label_count
        array[labels_out != max_label] = 0.

        resolution = 1. / self.grid_resolution

        grid = Voxelgrid(resolution)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = volume.translate
        bbox[:, 1] = bbox[:, 0] + resolution * volume.dims[0]

        grid.from_array(array, bbox)

        return grid
