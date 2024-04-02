import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.colors import LightSource
import seaborn as sns
import h5py


def visualize_ndvoxel(voxel, filename=None):
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
    ax = fig.gca(projection='3d')
    ax.set_axis_off()
    ax.voxels(voxel, facecolors=colors, lightsource=LightSource(azdeg=315, altdeg=45))
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    if filename:
        plt.savefig(fname=filename)
    else:
        plt.show()

def vis_voxel(voxel, save_image=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors='b', edgecolors='k')  # this can be slow
    plt.show()


def main(test=False):
    # shape_name = '14231'
    # shape_name = '14240'
    # shape_name = '14277'
    # shape_name = '14297'
    shape_name = '14300'
    dir = 'data/Lamp/' if not test else 'proj_log/pqnet-PartNet-Lamp_copy/results/rec-ckpt-1000-voxel-p0/'
    path = dir + shape_name + '.h5'
    with h5py.File(path, 'r') as fp:
        voxel = fp['voxel'][:] if test else fp['shape_voxel64'][:]
    # vis_voxel(voxel)
    filename = f'voxel/{shape_name}_data' if not test else f'voxel/{shape_name}_rec'
    visualize_ndvoxel(voxel, filename)


if __name__ == "__main__":
    main(test=True)
    main(test=False)