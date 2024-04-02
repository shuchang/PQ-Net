import h5py
import os
import glob
import numpy as np
from tqdm import tqdm
import json
import argparse
from scipy.spatial import KDTree
from pc_utils import read_ply
import numpy.linalg as LA


def chamfer_distance(a, b):
    tree = KDTree(b)
    dist_a = tree.query(a)[0]
    tree = KDTree(a)
    dist_b = tree.query(b)[0]
    return np.mean(dist_a) + np.mean(dist_b)

def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = LA.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    num_point, num_features = array1.shape
    dist = 0
    av_dist1 = array2samples_distance(array1, array2)
    av_dist2 = array2samples_distance(array2, array1)
    dist = dist + (av_dist1+av_dist2)
    return dist


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
        arr1 = (1 - arr1).astype(np.bool)
        arr2 = (1 - arr2).astype(np.bool)
    
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
    parser.add_argument('--metric', type=str, default='iou')
    parser.add_argument('--data', type=str, help='data directory', default='/Users/shuc/Desktop/course/PQ-Net/data/Lamp')
    parser.add_argument('--rec', type=str, help='reconstruction directory', default='/Users/shuc/Desktop/course/PQ-Net/proj_log/pqnet-PartNet-Lamp_copy/results/rec-ckpt-1000-voxel-p0')
    
    args = parser.parse_args()

    data_root = args.data if args.metric == 'iou' else args.data + '_pc'
    rec_root = args.rec if args.metric == 'iou' else args.rec + '_pc'

    shape_names = sorted(os.listdir(rec_root))
    shape_names = [name for name in shape_names if name.endswith('.h5')]
    pc_names = [name for name in shape_names if name.endswith('.ply')]
    total_valid_nums = 0
    iou = []
    cd = []
    for name in tqdm(shape_names):
        vol_path1 = os.path.join(rec_root, name)
        with h5py.File(vol_path1, "r") as fp:
            voxel1 = fp['voxel'][:]
        vol_path2 = os.path.join(data_root, name)
        with h5py.File(vol_path2, "r") as fp:
            voxel2 = fp['shape_voxel64'][:]
    
        iou.append(IoUarr(voxel1, voxel2, inside=1))

    for name in tqdm(pc_names):
        pc_path1 = os.path.join(rec_root, name)
        pc1 = read_ply(pc_path1)
        pc_path2 = os.path.join(data_root, name)
        pc2 = read_ply(pc_path2)

        cd.append(chamfer_distance_numpy(pc1, pc2))

    iou = np.array(iou)
    cd = np.array(cd)
    print(iou.mean())
    # print(cd.mean())


if __name__ == "__main__":
    main()