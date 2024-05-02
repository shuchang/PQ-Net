import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.colors import LightSource
import seaborn as sns
import h5py
from config import get_config
from util.utils import remkdir, ensure_dir


def visualize_ndvoxel(voxel, fname=None):
    """

    :param voxel:like (64,64,64) ndarray
    :return:
    """
    from mpl_toolkits.mplot3d import axes3d
    voxel = np.squeeze(voxel)
    if len(voxel.shape) == 4:
        voxel = voxel[0]

    color_num = voxel.max()
    current_palette = sns.color_palette(as_cmap=True)
    colors = np.empty(voxel.shape, dtype=object)
    for i in range(color_num):
        colors[voxel == i + 1] = current_palette[i]

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.voxels(voxel, facecolors=colors, lightsource=LightSource(azdeg=315, altdeg=45))
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    if fname:
        plt.savefig(fname=fname)
    else:
        plt.show()


def vis_voxel(voxel, save_image=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors='b', edgecolors='k')  # this can be slow
    plt.show()


def main(vis_rec):
    # create experiment config
    config = get_config('pqnet')('test')
    vis_dir = "{}/results/voxels_{}".format(config.exp_dir, config.module)
    # remkdir(vis_dir)
    ensure_dir(vis_dir)
    Lamp_shapes = ['14231', '14240','14277', '14297', '14300']
    Chair_shapes = ['692', '753', '762', '1282', '1284']

    # test_Lamp_shapes = ['15729', '16698']
    # test_Chair_shapes = ['36717', '38037']

    for idx in Chair_shapes:
        if not vis_rec:
            dir = "data/{}/".format(config.category)
            fname = "{}/{}_data".format(vis_dir, idx)
        else:
            dir = "{}/results/rec-ckpt-{}-{}-p{}/".format(
                config.exp_dir, config.ckpt, config.format, int(config.by_part)
                )
            fname = "{}/{}_rec".format(vis_dir, idx)

        path = "{}{}.h5".format(dir, idx)
        with h5py.File(path, 'r') as fp:
            voxel = fp['voxel'][:] if vis_rec else fp['shape_voxel64'][:]

        # vis_voxel(voxel)
        visualize_ndvoxel(voxel, fname)


if __name__ == "__main__":
    main(vis_rec=True)
    main(vis_rec=False)