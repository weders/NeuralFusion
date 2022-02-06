import torch

from copy import copy


def mse_fn(est, gt, mask=None):

    esti = copy(est)
    gti = copy(gt)

    if mask is not None:
        esti = esti * mask
        gti = gti * mask
        normalization = torch.sum(mask)
    else:
        normalization = torch.sum(torch.ones_like(est))

    se = torch.pow(esti - gti, 2)
    se = torch.sum(se)
    mse = se / normalization
    mse = mse.item()
    return mse


def mad_fn(est, gt, mask=None):

    esti = copy(est)
    gti = copy(gt)

    if mask is not None:
        esti = esti * mask
        gti = gti * mask
        normalization = torch.sum(mask)
    else:
        normalization = torch.sum(torch.ones_like(est))

    ad = torch.abs(esti - gti)
    ad = torch.sum(ad)
    mad = ad / normalization
    mad = mad.item()

    return mad


def iou_fn(est, gt, mask=None):

    esti = copy(est)
    gti = copy(gt)

    if mask is not None:
        esti = esti * mask
        gti = gti * mask

    occ_est = torch.where(esti < 0, torch.ones_like(esti),  torch.zeros_like(esti))
    occ_gt = torch.where(gti < 0, torch.ones_like(gti), torch.zeros_like(gti))

    intersection = torch.where((occ_est == 1) & (occ_gt == 1),
                               torch.ones_like(occ_est),
                               torch.zeros_like(occ_est))

    union = torch.where((occ_est == 1) | (occ_gt == 1),
                        torch.ones_like(occ_est),
                        torch.zeros_like(occ_est))

    return (intersection.sum() / union.sum()).item()


def acc_fn(est, gt, mask=None):

    esti = copy(est)
    gti = copy(gt)

    if mask is not None:
        esti = esti * mask
        gti = gti * mask

    occ_est = torch.where(esti < 0, torch.ones_like(esti), torch.zeros_like(esti))
    occ_gt = torch.where(gti < 0, torch.ones_like(gti), torch.zeros_like(gti))

    free_est = torch.where(esti > 0, torch.ones_like(esti), torch.zeros_like(esti))
    free_gt = torch.where(gti > 0, torch.ones_like(gti), torch.zeros_like(gti))

    tp = torch.where((occ_est == 1.) & (occ_gt == 1.),
                     torch.ones_like(occ_est),
                     torch.zeros_like(occ_est))
    tn = torch.where((free_est == 1.) & (free_gt == 1.),
                     torch.ones_like(free_est),
                     torch.zeros_like(free_est))

    tp = torch.sum(tp)
    tn = torch.sum(tn)

    norm = torch.where(esti != 0., torch.ones_like(esti), torch.zeros_like(esti))
    norm = torch.sum(norm)

    acc = (tp + tn) / norm
    acc = acc.item()

    return acc

def f1_fn(est, gt, mask=None):

    esti = copy(est)
    gti = copy(gt)

    if mask is not None:
        esti = esti * mask
        gti = gti * mask

    occ_est = torch.where(esti < 0, torch.ones_like(esti), torch.zeros_like(esti))
    occ_gt = torch.where(gti < 0, torch.ones_like(gti), torch.zeros_like(gti))

    free_est = torch.where(esti > 0, torch.ones_like(esti), torch.zeros_like(esti))
    free_gt = torch.where(gti > 0, torch.ones_like(gti), torch.zeros_like(gti))

    tp = torch.where((occ_est == 1.) & (occ_gt == 1.),
                     torch.ones_like(occ_est),
                     torch.zeros_like(occ_est))
    tn = torch.where((free_est == 1.) & (free_gt == 1.),
                     torch.ones_like(free_est),
                     torch.zeros_like(free_est))

    fn = torch.where((occ_est == 0.) & (occ_gt == 1.),
                     torch.ones_like(occ_est),
                     torch.zeros_like(occ_est))
    fp = torch.where((occ_est == 1.) & (occ_gt == 0.),
                     torch.ones_like(occ_est),
                     torch.zeros_like(occ_est))

    tp = torch.sum(tp)
    tn = torch.sum(tn)
    fn = torch.sum(fn)
    fp = torch.sum(fp)

    f1 = tp / (tp + 0.5 * (fp + fn))

    return f1

class Metrics():

    def __init__(self, metrics):

        self.metric_fns = metrics
        self.metric_values = {}
        self.metric_count = 0.

        for m in metrics.keys():

            self.metric_values[m] = 0.

    def update(self, est, gt, mask=None):

        for m in self.metric_fns.keys():
            self.metric_values[m] += self.metric_fns[m](est, gt, mask)


