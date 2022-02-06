import os
import sys
import numpy as np
import struct
import collections


def read_dat_groundtruth(path):
    with open(path, "rb") as fid:
        version = np.fromfile(fid, count=1, dtype=np.uint8)
        assert version == 1

        is_big_endian = np.fromfile(fid, count=1, dtype=np.uint8)
        assert (is_big_endian == 1 and sys.byteorder == "big") or \
               (is_big_endian == 0 and sys.byteorder == "little")

        uint_size = np.fromfile(fid, count=1, dtype=np.uint8)
        assert uint_size == 4

        elem_size = np.fromfile(fid, count=1, dtype=np.uint32)
        if elem_size == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported data type of size {}".format(elem_size))

        num_labels = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        width = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        height = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        depth = np.fromfile(fid, count=1, dtype=np.uint32)[0]

        num_elems = width * height * depth * num_labels
        assert num_elems > 0

        grid = np.fromfile(fid, count=num_elems, dtype=dtype)
        grid = grid.reshape(depth, height, width, num_labels)

        # TODO: check if this is the correct order
        grid = grid.transpose(2, 1, 0, 3)
        grid = np.squeeze(grid, axis=-1)
        grid = np.ascontiguousarray(grid)

        return grid


def read_groundtruth(room_path, numclasses):

    # Read the dat file
    groundtruth = read_dat_groundtruth(os.path.join(room_path, "GroundTruth.dat"))

    # Change labeling so that free space is the last
    groundtruth = np.squeeze(groundtruth)
    groundtruth -= 1
    groundtruth[groundtruth < 0] = numclasses - 1

    # Change to one hot encoding (ground truth prob)
    xres, yres, zres = np.shape(groundtruth)
    groundtruth = np.reshape(groundtruth, groundtruth.size)
    groundtruth_onehot = np.zeros((groundtruth.size, numclasses), dtype=np.float32)
    groundtruth_onehot[np.arange(groundtruth.size), groundtruth] = 1.0
    groundtruth_onehot = np.reshape(groundtruth_onehot, [xres, yres, zres, numclasses])

    return groundtruth


def read_camera_file(scene_path, width, height, camera_id):

    cam_file = os.path.join(scene_path, "cameras.txt")

    # get camera parameters from file
    fid = open(cam_file, 'r')
    line = fid.readlines()[camera_id]
    elems = list(map(lambda x: float(x.strip()), line.split(",")))
    elems = np.array(elems)
    
    # camera origin
    eye = elems[0:3]
    
    # compute camera coordinate system
    towards = -elems[3:6]
    up = -elems[6:9]

    cy = up
    cy /= np.linalg.norm(cy)
    cz = -towards
    cz /= np.linalg.norm(cz)
    cx = np.cross(cy, cz)
    
    # build rotation matrix
    rotation = np.hstack((cx, cy, cz))
    rotation = np.reshape(rotation, (3, 3))
    rotation = np.transpose(rotation)

    # build extrinsics
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = eye
    #extrinsics[0, :3] *= -1.
    #extrinsics[1, :3] *= -1.


    # camera intrinsic
    fov = elems[9:11]
    focal_length_x = width / (2 * np.tan(fov[0]))
    focal_length_y = height / (2 * np.tan(fov[1]))

    intrinsics = np.eye(3)
    intrinsics[0, 0] = focal_length_x
    intrinsics[1, 1] = focal_length_y
    intrinsics[0, 2] = width / 2
    intrinsics[1, 2] = height / 2

    return extrinsics, intrinsics


def read_projections(scene_path, width, height, camera_id, mode=None):

    cam_file = os.path.join(scene_path, "cameras.txt")

    # Initialize list of projection, rotation, position and intrinsics
    fid = open(cam_file, 'r')
    line = fid.readlines()[camera_id]

    elems = list(map(lambda x: float(x.strip()), line.split(",")))
    elems = np.array(elems)

    # Create rigth handed orthonormal basis
    cam_center =  elems[0:3] # camera position
    towards    = -elems[3:6] # pointing direction
    updir      = -elems[6:9] # up direction

    towards /= np.linalg.norm(towards)

    right = np.cross(towards, updir)
    right /= np.linalg.norm(right)

    updir = np.cross(right, towards)

    # Create camera rotation
    cam_rot = np.eye(3)
    cam_rot[0, :] = right
    cam_rot[1, :] = updir
    cam_rot[2, :] = -towards

    # Create camera translation: T = -RC
    cam_trans = -np.matmul(cam_rot, np.expand_dims(cam_center,axis=-1))

    # Camera extrinsics P = [R|T]
    cam_pose = np.concatenate([cam_rot, cam_trans], axis=-1)

    # Camera intrinsic
    fov = elems[9:11]

    focal_length_x = 0.5 * width / np.tan(fov[0])
    focal_length_y = 0.5 * height / np.tan(fov[1])

    intrinsics = np.eye(3)
    intrinsics[0, 0] = focal_length_x
    intrinsics[1, 1] = focal_length_y
    intrinsics[0, 2] = width/2
    intrinsics[1, 2] = height/2

    if mode == 'projection':
        projection = np.dot(intrinsics, cam_pose)
        return projection

    # Projection matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :4] = cam_pose  # proj = K[R|T]
    extrinsics = extrinsics.astype(np.float32)
    extrinsics = np.linalg.inv(extrinsics)

    #extrinsics[0, 3] -= 0.15

    return extrinsics, intrinsics, cam_rot, cam_center


class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
               "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline();
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def write_trajectory(traj, filename):
    with open(filename, 'w') as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(' '.join(map(str, x.metadata)) + '\n')
            f.write('\n'.join(
                ' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
            f.write('\n')


def read_camera(filename):

    intrinsics = []
    rotation = []
    translation = []

    with open(filename, 'rb') as file:
        for i, line in enumerate(file):
            if i < 3:
                row = [float(r) for r in
                       line.decode('utf-8').rstrip().split(' ')]
                intrinsics.append(row)
            elif i < 6:
                row = [float(r) for r in
                       line.decode('utf-8').rstrip().split(' ')]
                rotation.append(row)
                continue
            elif i == 6:
                row = [float(r) for r in
                       line.decode('utf-8').rstrip().split(' ')]
                translation.append(row)
            elif i == 7:
                resolution = [float(r) for r in line.decode('utf-8').rstrip().split(' ')]
                continue
            elif i == 8:
                imagefile = line.decode('utf-8').rstrip()
                continue
            else:
                depthfile = line.decode('utf-8').rstrip()

    rotation = np.asarray(rotation)
    translation = np.asarray(translation).T

    M = np.eye(3, 3)
    M[0, 0] = -1.
    M[1, 1] = -1.

    rotation = np.dot(M, rotation)
    translation = np.dot(M, translation)

    extrinsics = np.hstack((rotation, translation))
    extrinsics = np.vstack((extrinsics, np.asarray([0, 0, 0, 1.])))

    camera = {'intrinsics': np.asarray(intrinsics),
              'extrinsics': extrinsics,
              'resolution': (resolution[0], resolution[1]),
              'imagefile': imagefile,
              'depthfile': depthfile}

    return camera

def read_next_bytes(fid, num_bytes, format_char_sequence,
                    endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)

            cameras[camera_id] = params


    return cameras


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_images(path):

    images = {}

    with open(path, 'r') as file:
        for i, line in enumerate(file):
            if i % 2 == 0:
                if line[0] == '#':
                    continue

                elements = line.rstrip().split(' ')

                image_id = elements[0]

                qw = elements[1]
                qx = elements[2]
                qy = elements[3]
                qz = elements[4]

                tx = elements[5]
                ty = elements[6]
                tz = elements[7]

                camera_id = elements[8]

                name = elements[9]

                quaternion = np.asarray([float(qw),
                                         float(qx),
                                         float(qy),
                                         float(qz)])
                translation = np.asarray([float(tx),
                                          float(ty),
                                          float(tz)])

                images[str(image_id)] = {}
                images[image_id]['camera_id'] = camera_id
                images[image_id]['name'] = name
                images[image_id]['quaternion'] = quaternion
                images[image_id]['translation'] = translation

    return images

def read_cameras(path):
    cameras = {}

    with open(path, 'r') as file:
        for line in file:
            if line[0] == '#':
                continue

            # parse camera line
            elements = line.rstrip().split(' ')
            camera_id = elements[0]
            model = elements[1]
            width = float(elements[2])
            height = float(elements[3])
            fx = float(elements[4])
            fy = float(elements[5])
            px = float(elements[6])
            py = float(elements[7])

            # create camera entry
            cameras[camera_id] = {}
            cameras[camera_id]['model'] = model
            cameras[camera_id]['width'] = width
            cameras[camera_id]['height'] = height
            cameras[camera_id]['fx'] = fx
            cameras[camera_id]['fy'] = fy
            cameras[camera_id]['px'] = px
            cameras[camera_id]['py'] = py

    return cameras

import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        result = {}

        for key in sample.keys():
            if type(sample[key]) is np.ndarray:

                if key == 'image':
                    # swap color axis because
                    # numpy image: H x W x C
                    # torch image: C X H X W
                    image = sample[key].transpose((2, 0, 1))
                    image = torch.from_numpy(image)
                    result[key] = image.float()
                    continue

                result[key] = torch.from_numpy(sample[key]).float()

            else:
                result[key] = sample[key]

        return result
