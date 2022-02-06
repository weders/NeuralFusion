import os
import glob
import cc3d
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
from copy import copy

from graphics import Voxelgrid
from graphics.transform import compute_tsdf
import h5py

# from graphics.utils import extract_mesh_marching_cubes
# from graphics.visualization import plot_mesh, plot_voxelgrid

from scipy.ndimage.morphology import binary_dilation

from .utils.augmentation import *

from .utils.binvox_utils import read_as_3d_array


class ShapeNet(Dataset):

    def __init__(self,
                 root_dir,
                 obj=None,
                 model=None,
                 scene_list=None,
                 resolution=(240, 320),
                 transform=None,
                 noise_scale=0.05,
                 outlier_scale=3,
                 outlier_fraction=0.99,
                 grid_resolution=64,
                 repeat=0,
                 load_smooth=False):

        self.noise_scale = noise_scale
        self.root_dir = os.path.expanduser(root_dir)

        self.resolution = resolution
        self.xscale = resolution[0] / 480.
        self.yscale = resolution[1] / 640.

        self.transform = transform

        self.obj = obj
        self.model = model
        self.scene_list = scene_list

        self.noise_scale = noise_scale
        self.outlier_scale = outlier_scale
        self.outlier_fraction = outlier_fraction

        self.grid_resolution = grid_resolution

        self.repeat = repeat
        self.load_smooth = load_smooth

        self._load_frames()

    def _load_frames(self):

        if self.scene_list is None:
            # scene, obj = self.scene.split('/')
            path = os.path.join(self.root_dir, self.obj, self.model, 'data', '*.depth.png')
            files = glob.glob(path)

            self.frames = []

            for f in files:
                self.frames.append(f.replace('.depth.png', ''))

            self._scenes = [os.path.join(self.obj, self.model)]

        else:

            self.frames = []
            self._scenes = []

            with open(self.scene_list, 'r') as file:

                for line in file:
                    scene, obj = line.rstrip().split('\t')

                    path = os.path.join(self.root_dir, scene, obj, 'data', '*.depth.png')

                    files = glob.glob(path)

                    for f in files:
                        self.frames.append(f.replace('.depth.png', ''))

                    if os.path.join(scene, obj) not in self._scenes:
                        self._scenes.append(os.path.join(scene, obj))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):

        frame = self.frames[item]

        pathsplit = frame.split('/')
        sc = pathsplit[-4]
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
        sample['original_mask'] = copy(mask)
        gradient_mask = binary_dilation(mask, iterations=5)
        mask = binary_dilation(mask, iterations=8)
        sample['mask'] = mask
        sample['gradient_mask'] = gradient_mask

        depth[mask == 0] = 0

        sample['depth'] = depth
        sample['noisy_depth'] = add_kinect_noise(copy(depth), sigma_fraction=self.noise_scale)
        sample['noisy_depth_octnetfusion'] = add_depth_noise(copy(depth), noise_sigma=self.noise_scale, seed=42)
        sample['outlier_depth'] = add_outliers(copy(sample['noisy_depth_octnetfusion']),
                                               scale=self.outlier_scale,
                                               fraction=self.outlier_fraction)
        sample['sparse_depth'] = add_sparse_depth(copy(sample['noisy_depth_octnetfusion']),
                                                  percentage=0.001)
        sample['outlier_blob_depth'] = add_outlier_blobs(copy(sample['noisy_depth_octnetfusion']),
                                                         scale=self.outlier_scale,
                                                         fraction=self.outlier_fraction)

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
            if type(sample[key]) is not np.ndarray and type(sample[key]) is not str:
                sample[key] = np.asarray(sample[key])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self, scene, resolution=None):

        sc, obj = scene.split('/')

        if not self.load_smooth:


            # if self.grid_resolution == 25604530566	10fe40ebace4de15f457958925a36a51:
            #     filepath = os.path.join(self.root_dir, sc, obj, 'voxels', '*.{}.binvox')
            #     print(filepath)
            # else:
            #     filepath = os.path.join(self.root_dir, sc, obj, 'voxels', '*.{}.binvox'.format(self.grid_resolution))
            filepath = os.path.join(self.root_dir, sc, obj, 'voxels', '*.{}.binvox'.format(self.grid_resolution))

            filepath = glob.glob(filepath)[0]


            # filepath = os.path.join(self.root_dir, 'example', 'voxels', 'chair_0256.binvox')

            with open(filepath, 'rb') as file:
                volume = read_as_3d_array(file)

            scene = volume.data
            scene = scene.astype(np.int)

            labels_out = cc3d.connected_components(scene)  # 26-connected
            N = np.max(labels_out)
            max_label = 0
            max_label_count = 0
            valid_labels = []
            for segid in range(1, N + 1):
                extracted_image = labels_out * (labels_out == segid)
                extracted_image[extracted_image != 0] = 1
                label_count = np.sum(extracted_image)
                if label_count > max_label_count:
                    max_label = segid
                    max_label_count = label_count
                if label_count > 1000:
                    valid_labels.append(segid)

            for segid in range(1, N + 1):
                if segid not in valid_labels:
                    scene[labels_out == segid] = 0.
            if not resolution:
                resolution = 1. / self.grid_resolution
            else:
                resolution

            grid = Voxelgrid(resolution)
            bbox = np.zeros((3, 2))
            bbox[:, 0] = volume.translate
            bbox[:, 1] = bbox[:, 0] + resolution * volume.dims[0]

            grid.from_array(scene, bbox)

        else:

            bbox = np.zeros((3, 2))

            bbox[0, 0] = -0.5
            bbox[1, 0] = -0.5
            bbox[2, 0] = -0.5

            bbox[0, 1] = 0.5
            bbox[1, 1] = 0.5
            bbox[2, 1] = 0.5

            filepath = os.path.join(self.root_dir, sc, obj, 'smooth', 'model.hf5')

            with h5py.File(filepath, 'r') as file:
                volume = file['TSDF'][:]

            resolution = 1. / self.grid_resolution
            grid = Voxelgrid(resolution)
            grid.from_array(volume, bbox)

        return grid

    def get_tsdf(self, scene, resolution=None):

        sc, obj = scene.split('/')

        if self.grid_resolution == 256:
            filepath = os.path.join(self.root_dir, sc, obj, 'voxels', '*.binvox')
        else:
            filepath = os.path.join(self.root_dir, sc, obj, 'voxels', '*.{}.binvox'.format(self.grid_resolution))
        filepath = glob.glob(filepath)[0]

        # filepath = os.path.join(self.root_dir, 'example', 'voxels', 'chair_0256.binvox')

        with open(filepath, 'rb') as file:
            volume = read_as_3d_array(file)

        scene = volume.data
        scene = scene.astype(np.int)

        labels_out = cc3d.connected_components(scene)  # 26-connected
        N = np.max(labels_out)
        max_label = 0
        max_label_count = 0
        valid_labels = []
        for segid in range(1, N + 1):
            extracted_image = labels_out * (labels_out == segid)
            extracted_image[extracted_image != 0] = 1
            label_count = np.sum(extracted_image)
            if label_count > max_label_count:
                max_label = segid
                max_label_count = label_count
            if label_count > 10000:
                valid_labels.append(segid)

        for segid in range(1, N + 1):
            if segid not in valid_labels:
                scene[labels_out == segid] = 0.

        # computing tsdf
        dist1 = compute_tsdf(scene.astype(np.float64))
        dist1[dist1 > 0] -= 0.5
        dist2 = compute_tsdf(np.ones(scene.shape) - scene)
        dist2[dist2 > 0] -= 0.5
        # print(np.where(dist == 79.64923100695951))
        scene = np.copy(dist1 - dist2)

        if not resolution:
            resolution = 1. / self.grid_resolution
        else:

            step_size = int(self.grid_resolution / resolution)
            scene = scene[::step_size, ::step_size, ::step_size]

            resolution = 1. / resolution

        grid = Voxelgrid(resolution)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = volume.translate
        bbox[:, 1] = bbox[:, 0] + resolution * scene.shape[0]

        grid.from_array(scene, bbox)
        
        return grid

    @property
    def scenes(self):
        return self._scenes


if __name__ == '__main__':

    import matplotlib.pyplot as plt


    dataset = ShapeNet('/media/weders/HV620S/data/shape-net/processed',
                       '03001627',
                       '1007e20d5e811b308351982a6e40cf41',
                       grid_resolution=128)

    for f in dataset:
        plt.imshow(f['sparse_depth'])
        plt.show()
