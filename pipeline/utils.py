import torch

from torch.nn.functional import normalize


def create_canonical_view(points, eye, n_samples):

    directions = points - eye
    directions = normalize(directions, p=2, dim=2)

    canonical_view = [points]

    for i in range(1, n_samples + 1):
        p = points + i * 1. * directions
        pN = points - i * 1. * directions

        canonical_view.append(p)
        canonical_view.insert(0, pN)

    canonical_view = torch.stack(canonical_view, dim=2)

    return canonical_view, directions


def depth_routing(routing, batch, threshold):

    with torch.no_grad():
        output = routing.forward(batch['input'].unsqueeze_(0))
        batch['input'] = output[:, 0, :, :, ]
        batch['confidence'] = output[:, 1, :, :]

        confidence = torch.exp(-1. * batch['confidence'])

        batch['input'][confidence < threshold] = 0.
        batch['original_mask'][confidence < threshold] = 0.

    return batch


def unproject(depth, extrinsics, intrinsics, origin, resolution, device):

        b, h, w = depth.shape
        n_points = h * w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float)])

        # flatten grid coordinates and bring them to batch size
        xx = xx.contiguous().view(1, h * w, 1).repeat((b, 1, 1)).to(depth)
        yy = yy.contiguous().view(1, h * w, 1).repeat((b, 1, 1)).to(depth)
        zz = depth.contiguous().view(b, h * w, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone()

        # invert intrinsics matrix
        intrinsics_inv = intrinsics.inverse()

        homogenuous = torch.ones((b, 1, n_points)).to(depth)

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        points_g = (points_w - origin) / resolution

        return points_g
