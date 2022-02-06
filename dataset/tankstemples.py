import os
import torch
import cv2
import h5py

import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor, Compose
from skimage.io import imread
from copy import copy, deepcopy
from graphics import Voxelgrid

from .utils.read_write_model import read_model
from .utils.colmap import read_array
from .utils.augmentation import add_outliers

class TanksTemples(Dataset):

    def __init__(self, config):

        super(TanksTemples, self).__init__()

        self.root_dir = config.root_dir
        self.scene = config.scene
        self.colmap_path = os.path.join(self.root_dir,
                                        self.scene,
                                        self.scene.title(),
                                        'dense/0')

        self.resolution = (config.resy, config.resx)

        self.stereo_method = config.stereo_method

        self.__init_dataset()


    def __init_dataset(self):
        cameras, images, points3d = read_model(os.path.join(self.colmap_path,
                                                            'sparse'))

        self.cameras = cameras
        self.images = images
        self.points3d = points3d

        self.frames = []
        for k in self.cameras:
            if self.scene == 'caterpillar' and k in [316, 320]:
                continue
            self.frames.append(k)


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):

        sample = {}

        frame = self.frames[item]

        camera = self.cameras[frame]
        image = self.images[frame]

        image_name = image.name

        depth_path = os.path.join(self.colmap_path, 'stereo/depth_maps')
        depth_file = os.path.join(depth_path, '{}.{}.bin'.format(image_name, self.stereo_method))

        depth = read_array(depth_file)

        # get extrinsics
        rotation = image.qvec2rotmat()
        translation = image.tvec

        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation

        extrinsics = np.linalg.inv(extrinsics)

        # get intrinsics
        intrinsics = np.eye(3, 3)
        intrinsics[0, 0] = camera.params[0]
        intrinsics[1, 1] = camera.params[1]
        intrinsics[0, 2] = camera.params[2]
        intrinsics[1, 2] = camera.params[3]

        # resizing image

        # crop first
        centery = int(depth.shape[0] / 2)
        centerx = int(depth.shape[1] / 2)

        depth = depth[centery-360:centery+360, centerx-480:centerx+480]
        intrinsics[0, 2] = 480.
        intrinsics[1, 2] = 320.

        scale_y = depth.shape[0] / self.resolution[0]
        scale_x = depth.shape[1] / self.resolution[1]

        step_x = int(scale_x)
        step_y = int(scale_y)

        depth = depth[::step_y, :]
        depth = depth[:, ::step_x]

        # clamp depth
        depth[depth > 4.] = 0
        # get mask
        mask = np.ones_like(depth)
        mask[depth == 0.] = 0

        scaling = np.eye(3, 3)
        scaling[0, 0] = 1./scale_x
        scaling[1, 1] = 1./scale_y

        intrinsics = np.dot(scaling, intrinsics)

        sample['scene_id'] = self.scene
        sample['depth'] = depth.astype(np.float32)
        sample['outlier_depth'] = add_outliers(copy(depth), scale=0.3, fraction=0.9)
        sample['extrinsics'] = extrinsics.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)
        sample['original_mask'] = mask.astype(np.float32)

        return sample

    @property
    def scenes(self):
        return [self.scene]

    def get_grid(self, scene):

        sdf_file = os.path.join(self.root_dir, self.scene, '{}_tsdf_256.hf5'.format(self.scene))
        bbox_file = os.path.join(self.root_dir, self.scene, 'bbox.txt')

        with h5py.File(sdf_file, 'r') as file:
            data = file['tsdf'][:]
        # init voxelgrid
        grid = Voxelgrid(1./64.)

        # init bbox
        bbox = np.loadtxt(bbox_file)

        # create voxelgrid from array
        grid.from_array(data, bbox)

        print(grid.volume.shape)
        return grid



if __name__ == '__main__':

    import open3d as o3d

    from easydict import EasyDict
    from tqdm import tqdm

    def draw_registration_result(source, target, transformation):
        source_temp = deepcopy(source)
        target_temp = deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

    config = EasyDict()

    config.root_dir = '/media/weders/datasets/tankstemples/training'
    config.scene = 'truck'
    config.resx = 320
    config.resy = 240
    config.stereo_method = 'geometric'

    dataset = TanksTemples(config)

    color = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=1./64.,
                                                          sdf_trunc=0.06,
                                                          color_type=color)

    for frame in tqdm(dataset, total=len(dataset)):
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width=frame['depth'].shape[1],
                                                       height=frame['depth'].shape[0],
                                                       fx=frame['intrinsics'][0, 0],
                                                       fy=frame['intrinsics'][1, 1],
                                                       cx=frame['intrinsics'][0, 2],
                                                       cy=frame['intrinsics'][1, 2])

        rgb = o3d.geometry.Image(np.ones_like(frame['depth']))

        depth = frame['outlier_depth'].astype(np.float) * 1000.
        depth = depth.astype(np.uint16)
        depth = o3d.geometry.Image(depth)

        image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                   depth,
                                                                   depth_scale=1000,
                                                                   depth_trunc=4.,
                                                                   convert_rgb_to_intensity=False)

        extrinsics = np.linalg.inv(frame['extrinsics'].astype(np.float32))

        volume.integrate(image,
                         intrinsics,
                         extrinsics)



    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    point_cloud = volume.extract_point_cloud()

    min_bbox = point_cloud.get_min_bound()
    max_bbox = point_cloud.get_max_bound()

    print(min_bbox, max_bbox)

    min_bounds = np.asarray([[-2, -0.5, -2.1]]).reshape((3, 1))
    max_bounds = np.asarray([[3, 1.5, 2.4]]).reshape((3, 1))

    bbox_to_write = np.hstack((min_bounds, max_bounds))

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds,
                                               max_bounds)
    pc_cropped = point_cloud.crop(bbox)
    o3d.visualization.draw_geometries([pc_cropped])

    min_bounds = pc_cropped.get_min_bound().reshape((3, 1))
    max_bounds = pc_cropped.get_max_bound().reshape((3, 1))

    print(min_bounds)
    print(max_bounds)

    bbox = np.hstack((min_bounds, max_bounds))
    np.savetxt(os.path.join(dataset.root_dir, dataset.scene, 'bbox.txt'), bbox)
    bbox = np.loadtxt(os.path.join(dataset.root_dir, dataset.scene, 'bbox.txt'))

    grid = Voxelgrid(1./64., bbox, origin=bbox[:, 0])

    with h5py.File(os.path.join(dataset.root_dir, dataset.scene, '{}_tsdf_256.hf5'.format(dataset.scene)), 'w') as file:
        file.create_dataset('tsdf', data=grid.volume, shape=grid.volume.shape)

    print(grid.volume.shape)
    from graphics.voxelgrid import FeatureGrid
    feature_grid = FeatureGrid(resolution=1./64., bbox=bbox, origin=bbox[:, 0], n_features=9)
    print(feature_grid.data.shape)
   # pcd_target = volume.extract_point_cloud()
   # pcd_source = o3d.io.read_point_cloud(os.path.join(dataset.root_dir,
   #                                                   dataset.scene,
   #                                                   'point_cloud.ply'))

   # icp_initialization = np.eye(4, 4)
   # draw_registration_result(pcd_source, pcd_target, icp_initialization)
   # reg_p2p = o3d.pipelines.registration.registration_icp(pcd_source, pcd_target, 0.2, icp_initialization)
   ## np.savetxt(os.path.join(dataset.root_dir,
   ##                         dataset.scene,
   ##                         'gt_to_colmap.txt'), reg_p2p.transformation)

   # print(reg_p2p.transformation)
   # draw_registration_result(pcd_source, pcd_target, reg_p2p.transformation)
