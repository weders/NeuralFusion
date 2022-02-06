import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import torch
import io

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


from utils.visualization import add_mesh_to_tb

from graphics.utils import extract_mesh_marching_cubes


def get_centered_cmap(matrix):
    ax = sns.heatmap(matrix, center=0.)
    cmap = ax.collections[0].cmap
    return cmap


def plot_sdf(sdf):

    figure = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(sdf, center=0.)
    return figure


def plot_feature_map(feature_map):
    figure = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(feature_map, center=0.)
    return figure


def plot_loss(matrix):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(matrix)
    plt.set_cmap('jet')
    plt.colorbar()
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to PyTorch tensor
    image = Image.open(buf)
    transform = ToTensor()
    image = transform(image)

    return image


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0,
                    name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = torch.min(value) if vmin is None else vmin
    vmax = torch.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    # value = value.squeeze_(-1)

    h, w = value.shape

    # quantize
    indices = torch.round(value * 255).int()

    # gather
    if isinstance(cmap, str):
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    else:
        cm = cmap

    colors = torch.Tensor(cm.colors).float()

    indices = indices.view(indices.shape[0] * indices.shape[1])
    indices = indices.long()

    value = colors[indices, :3]
    value = value.view(1, h, w, 3)
    value = value.permute(0, -1, 1, 2)

    del colors, indices, vmin, vmax

    return value


def visualize_iou(tag, est, gt, writer, epoch):

    # generate iou image
    intersection = (est < 0) & (gt < 0)
    fp = (est < 0) & (gt >= 0)
    fn = (est >= 0) & (gt < 0)

    image = np.stack((fp, intersection, fn))

    writer.add_image(tag, image, global_step=epoch)


def visualize_sdf(tag, image, writer, epoch):
    """
    Function to visualize sdf plot
    """
    figure = plot_sdf(image)
    figure = plot_to_image(figure)
    writer.add_image(tag, figure, global_step=epoch)


def visualize_l1(tag, est, gt, writer, epoch):

    loss = np.abs(est - gt)

    figure = plot_loss(loss)
    image = plot_to_image(figure)
    writer.add_image(tag, image, global_step=epoch)


def visualize_l2(tag, est, gt, writer, epoch):

    loss = np.power(est - gt, 2)
    figure = plot_loss(loss)
    image = plot_to_image(figure)
    writer.add_image(tag, image, global_step=epoch)


def visualize_feature_map(tag, feature_map, writer, epoch):
    figure = plot_feature_map(feature_map)
    image = plot_to_image(figure)
    writer.add_image(tag, image, global_step=epoch)


def visualize_normal_map(tag, normal_map_est, normal_map_gt, writer, epoch):

    normal_map_est = 0.5 * normal_map_est + 0.5
    normal_map_gt = 0.5 * normal_map_gt + 0.5

    normal_map = torch.cat([normal_map_est, normal_map_gt], dim=-1)
    writer.add_image(tag, normal_map, global_step=epoch)


def visualize_validation_grid(scene_id, est, occ, gt, writer, epoch, features=None, normals_est=None, normals_gt=None):

    res = est.shape[-1]

    est_grid = est.view(res, res, res).cpu().detach().numpy()
    occ_grid = occ.view(res, res, res).cpu().detach().numpy()
    gt_grid = gt.view(res, res, res).cpu().detach().numpy()


    # visualize mesh
    est_mesh = extract_mesh_marching_cubes(est_grid, level=-1.e-08)
    gt_mesh = extract_mesh_marching_cubes(gt_grid, level=-1.e-08)
    occ_mesh = extract_mesh_marching_cubes(occ_grid, level=-1.e-08)

    if est_mesh:
        add_mesh_to_tb(gt_mesh, 'Validation/Groundtruth/{}'.format(scene_id), epoch, writer)
        add_mesh_to_tb(est_mesh, 'Validation/Prediction/{}'.format(scene_id), epoch, writer)
    if occ_mesh:
        add_mesh_to_tb(occ_mesh, 'Validation/Occupancy/{}'.format(scene_id), epoch, writer)



    # get slices
    index = int(res / 2)

    slicex_est = est_grid[index, :, :]
    slicex_gt = gt_grid[index, :, :]

    slicey_est = est_grid[:, index, :]
    slicey_gt = gt_grid[:, index, :]

    slicez_est = est_grid[:, :, index]
    slicez_gt = gt_grid[:, :, index]

    # visualize iou
    visualize_iou('Validation/ioux/{}'.format(scene_id), slicex_est, slicex_gt, writer, epoch)
    visualize_iou('Validation/iouy/{}'.format(scene_id), slicey_est, slicey_gt, writer, epoch)
    visualize_iou('Validation/iouz/{}'.format(scene_id), slicez_est, slicez_gt, writer, epoch)

    # visualize sdf
    visualize_sdf('Validation/sdfx/{}'.format(scene_id), slicex_est, writer, epoch)
    visualize_sdf('Validation/sdfy/{}'.format(scene_id), slicey_est, writer, epoch)
    visualize_sdf('Validation/sdfz/{}'.format(scene_id), slicez_est, writer, epoch)

    # visualize loss l1
    visualize_l1('Validation/l1x/{}'.format(scene_id), slicex_est, slicex_gt, writer, epoch)
    visualize_l1('Validation/l1y/{}'.format(scene_id), slicey_est, slicey_gt, writer, epoch)
    visualize_l1('Validation/l1z/{}'.format(scene_id), slicez_est, slicez_gt, writer, epoch)

    # visualize loss l2
    visualize_l2('Validation/l2x/{}'.format(scene_id), slicex_est, slicex_gt, writer, epoch)
    visualize_l2('Validation/l2y/{}'.format(scene_id), slicey_est, slicey_gt, writer, epoch)
    visualize_l2('Validation/l2z/{}'.format(scene_id), slicez_est, slicez_gt, writer, epoch)

    if normals_est is not None:

        normals_gt = normals_gt.cpu().detach()
        normals_est = normals_est.cpu().detach()

        normal_slicex_est = normals_est[0, :, index, :, :]
        normal_slicex_gt = normals_gt[0, :, index, :, :]


        visualize_normal_map('Validation/normalx/{}'.format(scene_id),
                             normal_slicex_est,
                             normal_slicex_gt,
                             writer,
                             epoch)

        normal_slicey_est = normals_est[0, :, :, index, :]
        normal_slicey_gt = normals_gt[0, :, :, index, :]

        visualize_normal_map('Validation/normaly/{}'.format(scene_id),
                             normal_slicey_est,
                             normal_slicey_gt,
                             writer,
                             epoch)

        normal_slicez_est = normals_est[0, :, :, :, index]
        normal_slicez_gt = normals_gt[0, :, :, :, index]

        visualize_normal_map('Validation/normalz/{}'.format(scene_id),
                             normal_slicez_est,
                             normal_slicez_gt,
                             writer,
                             epoch)

    if features is not None:

        features = features.cpu().detach().numpy()

        n_features = features.shape[1]

        feature_grid = features.cpu()

        res = features.shape[2]

        # get slices
        index = int(res / 2)

        for i in range(n_features):

            feature_mapx = feature_grid[0, i, index, :, :]
            feature_mapy = feature_grid[0, i, :, index, :]
            feature_mapz = feature_grid[0, i, :, :, index]

            visualize_feature_map('Validation/featuremapx{}/{}'.format(i, scene_id),
                                  feature_mapx,
                                  writer,
                                  epoch)

            visualize_feature_map('Validation/featuremapy{}/{}'.format(i, scene_id),
                                  feature_mapy,
                                  writer,
                                  epoch)

            visualize_feature_map('Validation/featuremapz{}/{}'.format(i, scene_id),
                                  feature_mapz,
                                  writer,
                                  epoch)


def visualize_training_grid(scene_id, est, occ, gt, writer, epoch, features=None,
                            normals_est=None, normals_gt=None):

    res = est.shape[-1]

    est_grid = est.view(res, res, res).cpu().detach().numpy()
    gt_grid = gt.view(res, res, res).cpu().detach().numpy()
    occ_grid = occ.view(res, res, res).cpu().detach().numpy()

    # visualize mesh
    est_mesh = extract_mesh_marching_cubes(est_grid, level=-1.e-08)
    gt_mesh = extract_mesh_marching_cubes(gt_grid, level=-1.e-08)
    occ_mesh = extract_mesh_marching_cubes(occ_grid, level=-1.e-08)

    if est_mesh:
        add_mesh_to_tb(gt_mesh, 'Training/Groundtruth/{}'.format(scene_id), epoch, writer)
        add_mesh_to_tb(est_mesh, 'Training/Prediction/{}'.format(scene_id), epoch, writer)
    if occ_mesh:
        add_mesh_to_tb(occ_mesh, 'Training/Occupancy/{}'.format(scene_id), epoch, writer)

    # get slices
    index = int(res / 2)

    slicex_est = est_grid[index, :, :]
    slicex_gt = gt_grid[index, :, :]

    slicey_est = est_grid[:, index, :]
    slicey_gt = gt_grid[:, index, :]

    slicez_est = est_grid[:, :, index]
    slicez_gt = gt_grid[:, :, index]

    # visualize iou
    visualize_iou('Training/ioux/{}'.format(scene_id), slicex_est, slicex_gt, writer, epoch)
    visualize_iou('Training/iouy/{}'.format(scene_id), slicey_est, slicey_gt, writer, epoch)
    visualize_iou('Training/iouz/{}'.format(scene_id), slicez_est, slicez_gt, writer, epoch)

    # visualize sdf
    visualize_sdf('Training/sdfx/{}'.format(scene_id), slicex_est, writer, epoch)
    visualize_sdf('Training/sdfy/{}'.format(scene_id), slicey_est, writer, epoch)
    visualize_sdf('Training/sdfz/{}'.format(scene_id), slicez_est, writer, epoch)

    # visualize loss l1
    visualize_l1('Training/l1x/{}'.format(scene_id), slicex_est, slicex_gt, writer, epoch)
    visualize_l1('Training/l1y/{}'.format(scene_id), slicey_est, slicey_gt, writer, epoch)
    visualize_l1('Training/l1z/{}'.format(scene_id), slicez_est, slicez_gt, writer, epoch)

    # visualize loss l2
    visualize_l2('Training/l2x/{}'.format(scene_id), slicex_est, slicex_gt, writer, epoch)
    visualize_l2('Training/l2y/{}'.format(scene_id), slicey_est, slicey_gt, writer, epoch)
    visualize_l2('Training/l2z/{}'.format(scene_id), slicez_est, slicez_gt, writer, epoch)

    if normals_est is not None:

        normals_gt = normals_gt.cpu().detach()
        normals_est = normals_est.cpu().detach()

        normal_slicex_est = normals_est[0, :, index, :, :]
        normal_slicex_gt = normals_gt[0, :, index, :, :]

        visualize_normal_map('Training/normalx/{}'.format(scene_id),
                             normal_slicex_est,
                             normal_slicex_gt,
                             writer,
                             epoch)

        normal_slicey_est = normals_est[0, :, :, index, :]
        normal_slicey_gt = normals_gt[0, :, :, index, :]

        visualize_normal_map('Training/normaly/{}'.format(scene_id),
                             normal_slicey_est,
                             normal_slicey_gt,
                             writer,
                             epoch)

        normal_slicez_est = normals_est[0, :, :, :, index]
        normal_slicez_gt = normals_gt[0, :, :, :, index]

        visualize_normal_map('Training/normalz/{}'.format(scene_id),
                             normal_slicez_est,
                             normal_slicez_gt,
                             writer,
                             epoch)

    if features is not None:

        features = features.cpu().detach().numpy()

        n_features = features.shape[1]

        res = features.shape[2]

        # get slices
        index = int(res / 2)

        feature_grid = features.cpu()

        for i in range(n_features):

            feature_mapx = feature_grid[0, i, index, :, :]
            feature_mapy = feature_grid[0, i, :, index, :]
            feature_mapz = feature_grid[0, i, :, :, index]

            visualize_feature_map('Training/featuremapx{}/{}'.format(i, scene_id),
                                  feature_mapx,
                                  writer,
                                  epoch)

            visualize_feature_map('Training/featuremapy{}/{}'.format(i, scene_id),
                                  feature_mapy,
                                  writer,
                                  epoch)

            visualize_feature_map('Training/featuremapz{}/{}'.format(i, scene_id),
                                  feature_mapz,
                                  writer,
                                  epoch)


