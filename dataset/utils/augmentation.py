import numpy as np
import random
import time


import cv2

from scipy.ndimage.filters import median_filter, maximum_filter, uniform_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import generate_binary_structure

def add_kinect_noise(depth, sigma_fraction=0.05):

    r = np.random.uniform(0., 1., depth.shape)
    sign = np.ones(depth.shape)
    sign[r < 0.5] = -1.0
    sigma = sigma_fraction*depth
    magnitude = sigma*(1.0 - np.exp(-0.5*np.power(r, 2)))
    depth += sign*magnitude
    depth[depth < 0] = 0.
    return depth


def add_axial_noise(x, std=0.05, depth_dependency=False, radial_dependency=False):

    if radial_dependency is False and depth_dependency is False:

        x += np.random.normal(0, scale=std)
        return x

    if depth_dependency:

        sigma = 0.0012 + 0.0019*np.power((x - 0.4), 2)
        x += np.random.normal(0, scale=sigma)
        return x


def add_random_zeros(x, p=0.9):

    mask = np.random.uniform(0, 1, x.shape)
    mask[mask >= p] = 0.0
    mask[mask > 0.0] = 1.0

    return np.multiply(x, mask)


def add_lateral_noise(x, focal_length=557, method='gaussian'):

    pixels = np.arange(-int(x.shape[1]/2), int(x.shape[1]/2), dtype=np.int32)
    theta = np.arctan(pixels/focal_length)

    sigma_l = 0.8 + 0.035*theta/(np.pi/2. - theta)

    x += np.random.normal(0, scale=sigma_l)
    return x


def add_depth_noise(depthmaps, noise_sigma, seed):

    # add noise
    if noise_sigma > 0:
        random.seed(time.process_time())
        np.random.seed(int(time.process_time()))
        sigma = noise_sigma
        noise = np.random.normal(0, 1, size=depthmaps.shape).astype(np.float32)
        depthmaps = depthmaps + noise * sigma * depthmaps

    return depthmaps


def add_lateral_and_axial_noise(x, focal_length):

    pixels = np.arange(-int(x.shape[1] / 2), int(x.shape[1] / 2), dtype=np.int32)
    theta = np.arctan(pixels / focal_length)

    sigma = 0.0012 + 0.0019*(x - 0-4)**2 + 0.0001/np.sqrt(x)*(theta**2)/(np.pi/2 - theta)**2

    x += np.random.normal(0, scale=sigma)
    return x


def add_outliers(x, scale=5, fraction=0.99):

    # check for invalid data points
    x[x < 0.] = 0.


    random.seed(time.process_time())
    np.random.seed(int(time.process_time()))

    # filter with probability:
    mask = np.random.uniform(0, 1, x.shape)
    mask[mask >= fraction] = 1.0
    mask[mask < fraction] = 0.0
    mask[x == 0.] = 0.

    outliers = np.random.normal(0, scale=scale, size=x.shape)
    x += np.multiply(outliers, mask)


    x[x < 0.] = 0.

    return x


def add_sparse_depth(x, percentage=0.1):

    # check for invalid data points
    x[x < 0.] = 0.

    random.seed(time.process_time())
    np.random.seed(int(time.process_time()))

    # filter with probability:
    mask = np.random.uniform(0, 1, x.shape)
    mask[mask < percentage] = -1
    mask[mask >= percentage] = 0.
    mask[x == 0.] = 0.

    x[mask == 0.] = 0.

    return x

def add_gradient_noise(x, xorig):

    gx = cv2.Sobel(xorig, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(xorig, cv2.CV_64F, 0, 1, ksize=5)

    grad = np.sqrt(gx ** 2 + gy ** 2)

    mask = np.zeros_like(grad)
    mask[grad > 1000] = 1.
    mask = binary_dilation(mask, iterations=4)

    noise = np.random.normal(0, 1, size=x.shape).astype(np.float32)

    x += np.multiply(mask, noise * 0.03 * x)

    return x


def add_outlier_blobs(x, scale=5, fraction=0.9, starti=1, endi=4):
    # check for invalid data points
    x[x < 0.] = 0.

    random.seed(time.process_time())
    np.random.seed(int(time.process_time()))

    # filter with probability:
    mask = np.random.uniform(0, 1, x.shape)
    mask[mask >= fraction] = 1.0
    mask[mask < fraction] = 0.0
    mask[x == 0.] = 0.



    for i in range(starti, endi):
        # filter with probability:
        mask = np.random.uniform(0, 1, x.shape)
        mask[mask <= (1. - fraction) / 3.] = -1.0
        mask[mask > (1. - fraction) / 3.] = 0.0
        mask[mask == -1.] = 1.
        mask[x == 0.] = 0.


        # dilation
        mask = binary_dilation(mask, iterations=i).astype(np.float)

        outliers = np.random.normal(0, scale=scale, size=x.shape)

        x += np.multiply(outliers, mask)

    x[x < 0.] = 0.

    return x


def add_noise_heuristic(x, xclean, scale=3., fraction=0.98):
    # check for invalid data points
    x[x < 0.] = 0.
    x_orig = np.copy(x)

    random.seed(time.process_time())
    np.random.seed(int(time.process_time()))

    gx = cv2.Sobel(xclean, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(xclean, cv2.CV_64F, 0, 1, ksize=5)

    grad = np.sqrt(gx ** 2 + gy ** 2)

    norm = np.count_nonzero(grad)
    thresh = np.sum(grad) / norm

    grad_mask = np.zeros_like(grad)
    grad_mask[grad > thresh] = 1.
    grad_mask = binary_erosion(grad_mask, iterations=1)

    # print(np.sum(grad_mask))

    for i in range(2, 5):

        mask = np.zeros_like(x)

        # filter with probability:
        sampler = np.random.uniform(0, 1, x.shape)
        mask[sampler <= (1. - fraction) / 3.] = 1.0

        mask[x == 0.] = 0.
        mask[grad_mask == 0.] = 0.

        # dilation
        mask = binary_dilation(mask, iterations=i).astype(np.float)

        outliers = np.random.normal(0, scale=scale, size=(15, 20))
        outliers = np.repeat(outliers, 16, axis=1)
        outliers = np.repeat(outliers, 16, axis=0)

        x += np.multiply(outliers, mask)
   
    x[x < 0.] = 0.

    return x
