import torch

import numpy as np

import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt

def add_mesh_to_tb(mesh, name, epoch, writer):
    (x, y, z) = (torch.Tensor(mesh['vertex'][t]) for t in ('x', 'y', 'z'))
    x = x.unsqueeze_(1)
    y = y.unsqueeze_(1)
    z = z.unsqueeze_(1)
    faces = torch.Tensor(mesh['face']['vertex_indices']).unsqueeze_(0)
    vertices = torch.cat((x, y, z), dim=1).unsqueeze_(0)
    colors = 150 * torch.ones_like(vertices)
    writer.add_mesh(name, vertices, colors=colors, faces=faces, global_step=epoch)


def add_normal_colored_mesh_to_tb(mesh, normals, epoch, writer):

    (x, y, z) = (torch.Tensor(mesh['vertex'][t]) for t in ('x', 'y', 'z'))
    x = x.unsqueeze_(1)
    y = y.unsqueeze_(1)
    z = z.unsqueeze_(1)

    x_rounded = torch.floor(x).long()
    y_rounded = torch.floor(y).long()
    z_rounded = torch.floor(z).long()

    slice = normals[0:2, :, :, 32]

    x, y = np.arange(0, 64), np.arange(0, 64)

    fig, ax = plt.subplots()
    q = ax.quiver(x, y, slice[0], slice[1])

    plt.show()

    # normal_directions = normals[:, x_rounded, y_rounded, z_rounded]
    #
    # faces = torch.Tensor(mesh['face']['vertex_indices']).unsqueeze_(0)
    # vertices = torch.cat((x, y, z), dim=1).unsqueeze_(0)
    #
    # colors = torch.floor(normal_directions * 255.).int().squeeze_(-1)
    #
    # writer.add_mesh('gt_normal_mesh', vertices, colors=colors, faces=faces, global_step=epoch)


def add_latent_embedding_to_tb(features, groundtruth, writer, epoch):
    _, ns, xs, ys, zs = features.shape

    embedding = features.view(xs * ys * zs, ns)
    labels = torch.sign(groundtruth).view(xs * ys * zs, 1)

    n_free = torch.sum(labels + torch.ones_like(labels)) / 2.
    n_occ = torch.abs(torch.sum(labels - torch.ones_like(labels))) / 2.

    w_free = n_occ / (n_free + n_occ)
    w_occ = n_free / (n_free + n_occ)

    weights = torch.where(labels > 0,
                          w_free * torch.ones_like(labels),
                          w_occ * torch.ones_like(labels))

    weights = weights / torch.sum(weights)
    weights = weights.squeeze_(-1).cpu().detach().numpy()

    points = np.arange(0, xs * ys * zs)

    indices = np.random.choice(points,
                               10000,
                               replace=False,
                               p=weights)

    embedding = embedding[indices, :]
    labels = labels[indices, :]

    writer.add_embedding(embedding, labels, global_step=epoch)



