import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse
from pc_utils import read_ply
import torch


def scale_to_unit_sphere(points, center=None):
    """
    scale point clouds into a unit sphere
    :param points: (n, 3) numpy array
    :return:
    """
    if center is None:
        midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
        # midpoints = np.mean(points, axis=0)
    else:
        midpoints = np.asarray(center)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    return points


def distChamfer(a, b):
    # Borrowed from PointFlow
    # https://github.com/stevenygd/PointFlow/blob/5caaf0be4e1ef5c7a8d4933092e9f3d0ac1b284e/metrics/evaluation_metrics.py#L11
    x, y = torch.tensor(a).unsqueeze(0), torch.tensor(b).unsqueeze(0)
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(x).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def IoUarr(arr1, arr2, inside=0, reduce_mean=True):
    """Calculate IoU for two batched input binary arrays. Default 0 for inside, 1 for outside.
    :param arr1: ndarray. (B, D1, D2, ...)
    :param arr2: ndarray. (B, D1, D2, ...)
    :param inside: int. Inside label, 0 or 1.
    :param reduce_mean: Boolean. Reduce mean over batch.
    :return:
    """
    if not arr1.shape == arr2.shape:
        raise ValueError("Two input arrays should be of equal size.")

    if not inside == 1:
        arr1 = (1 - arr1).astype(bool)
        arr2 = (1 - arr2).astype(bool)
    
    axes = tuple(range(1, len(arr1.shape)))
    intersection = np.sum(np.logical_and(arr1, arr2), axes)
    union = np.sum(np.logical_or(arr1, arr2), axes)
    iou = intersection / union

    if reduce_mean:
        iou = iou[~np.isnan(iou)]
        iou = np.mean(iou)

    return iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, choices=['iou', 'cd'], default='cd')
    # parser.add_argument('--data', type=str, help='data directory', default='/home/chang/projects/PQ-Net/data/Lamp')
    parser.add_argument('--data', type=str, help='data directory', default='/home/chang/projects/PQ-Net/data/Chair')
    # parser.add_argument('--rec', type=str, help='reconstruction directory', default='/home/chang/projects/PQ-Net/proj_log/pqnet-PartNet-Lamp/results/rec-ckpt-1000-voxel-p1-transformer')
    parser.add_argument('--rec', type=str, help='reconstruction directory', default='/home/chang/projects/PQ-Net/proj_log/pqnet-PartNet-Chair/results/rec-ckpt-latest-voxel-p1')
    
    args = parser.parse_args()

    data_root = args.data if args.metric == 'iou' else args.data + '_pc'
    rec_root = args.rec if args.metric == 'iou' else args.rec + '_pc'
    shape_names = sorted(os.listdir(rec_root))

    if args.metric == 'iou':
        vox_names = [name for name in shape_names if name.endswith('.h5')]
        iou = []

        for name in tqdm(vox_names):
            vol_path1 = os.path.join(rec_root, name)
            with h5py.File(vol_path1, "r") as fp:
                voxel1 = fp['voxel'][:]
            vol_path2 = os.path.join(data_root, name)
            with h5py.File(vol_path2, "r") as fp:
                voxel2 = fp['shape_voxel64'][:]
            iou.append(IoUarr(voxel1, voxel2, inside=1))

        iou = np.array(iou)
        print(iou.mean())

    elif args.metric == 'cd':
        torch.set_printoptions(precision=5)
        pc_names = [name for name in shape_names if name.endswith('.ply')]
        cd = []
    
        for name in tqdm(pc_names):
            pc_path1 = os.path.join(rec_root, name)
            pc1 = scale_to_unit_sphere(read_ply(pc_path1))
            pc_path2 = os.path.join(data_root, name)
            pc2 = scale_to_unit_sphere(read_ply(pc_path2))
            dl, dr = distChamfer(pc1, pc2)
            cd.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

        print(torch.cat(cd).mean())

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()