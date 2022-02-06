import torch


def unproject(depth, extrinsics, intrinsics, origin, resolution):

    b, h, w = depth.shape
    n_points = h * w

    # generate frame meshgrid
    xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float),
                             torch.arange(w, dtype=torch.float)])

    # flatten grid coordinates and bring them to batch size
    xx = xx.contiguous().view(1, h * w, 1).repeat((b, 1, 1))
    yy = yy.contiguous().view(1, h * w, 1).repeat((b, 1, 1))
    zz = depth.contiguous().view(b, h * w, 1)

    # generate points in pixel space
    points_p = torch.cat((yy, xx, zz), dim=2).clone()

    # invert intrinsics matrix
    intrinsics_inv = intrinsics.inverse()

    homogenuous = torch.ones((b, 1, n_points))

    # transform points from pixel space to camera space to world space (p->c->w)
    points_p[:, :, 0] *= zz[:, :, 0]
    points_p[:, :, 1] *= zz[:, :, 0]
    points_c = torch.matmul(intrinsics_inv,
                            torch.transpose(points_p, dim0=1, dim1=2))
    points_c = torch.cat((points_c, homogenuous), dim=1)
    points_w = torch.matmul(extrinsics[:3], points_c)
    points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

    points_g = (points_w - origin) / resolution

    return points_g


def interpolate(points):
    """
    Method to compute the interpolation indices and weights for trilinear
    interpolation
    """

    n = points.shape[0]

    # get indices
    indices = torch.floor(points)

    # compute interpolation distance
    df = torch.abs(points - indices)

    # get interpolation indices
    xx, yy, zz = torch.meshgrid([torch.arange(0, 2),
                                 torch.arange(0, 2),
                                 torch.arange(0, 2)])

    xx = xx.contiguous().view(8)
    yy = yy.contiguous().view(8)
    zz = zz.contiguous().view(8)

    shift = torch.stack([xx, yy, zz], dim=1)

    if points.get_device() >= 0:
        shift = shift.to(points.get_device())

    # reshape
    shift = shift.unsqueeze_(0)
    indices = indices.unsqueeze_(1)

    # compute indices
    indices = indices + shift

    # init weights
    weights = torch.zeros_like(indices).sum(dim=-1)

    # compute weights
    weights[:, 0] = (1 - df[:, 0]) * (1 - df[:, 1]) * (1 - df[:, 2])
    weights[:, 1] = (1 - df[:, 0]) * (1 - df[:, 1]) * df[:, 2]
    weights[:, 2] = (1 - df[:, 0]) * df[:, 1] * (1 - df[:, 2])
    weights[:, 3] = (1 - df[:, 0]) * df[:, 1] * df[:, 2]
    weights[:, 4] = df[:, 0] * (1 - df[:, 1]) * (1 - df[:, 2])
    weights[:, 5] = df[:, 0] * (1 - df[:, 1]) * df[:, 2]
    weights[:, 6] = df[:, 0] * df[:, 1] * (1 - df[:, 2])
    weights[:, 7] = df[:, 0] * df[:, 1] * df[:, 2]

    weights = weights.unsqueeze_(-1)

    return indices, weights


def get_neighbourhood_indices(indices, size=(3, 3, 3)):

    # neighbourhood needs to be symmetric
    assert size[0] % 2 != 0
    assert size[0] == size[1] and size[0] == size[2]

    indices = torch.floor(indices)

    start = -(size[0] // 2)
    end = (size[0] // 2) + 1

    xx, yy, zz = torch.meshgrid(torch.arange(start, end),
                                torch.arange(start, end),
                                torch.arange(start, end))

    xx = xx.contiguous().view(size[0]**3)
    yy = yy.contiguous().view(size[0]**3)
    zz = zz.contiguous().view(size[0]**3)

    neighbourhood = torch.stack((xx, yy, zz), dim=1)

    if indices.get_device() >= 0:
        neighbourhood = neighbourhood.to(indices.get_device())

    return neighbourhood