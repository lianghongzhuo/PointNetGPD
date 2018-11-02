# import math
import numpy as np
# from scipy import linalg
# from scipy import spatial
import sys
import os
import logging
# import matplotlib as mp
# import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.mlab import griddata
import pylab as plt

# import IPython
from meshpy.sdf_file import SdfFile
from meshpy.obj_file import ObjFile
from dexnet.grasping import GraspableObject3D

# DEF_SDF_FILE = 'textured.sdf'
DEF_SDF_FILE = '/home/liang/dataset/ycb_meshes_google/objects/003_cracker_box/google_512k/textured.sdf'

# SMALL_SDF_FILE = '../Teapot small N290514.sdf'


class SDF(object):
    def __init__(self, dims, origin, dx, sdf_values):
        self.dims = dims
        self.origin = origin
        self.dx = dx
        self.sdf_values = sdf_values


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/')+1:
            file_list.append(root)
        # print(dirs)  # all the directories in current path
        # print(files)  # all the files in current path
    file_list.sort()
    return file_list


def read_sdf(filename):
    """
    Returns a 3d numpy array of SDF values from the input file
    """
    try:
        f = open(filename, 'r')
    except IOError:
        logging.error('Failed to open sdf file: %s' % filename)
        return None
    dim_str = f.readline()
    origin_str = f.readline()
    dx_str = f.readline()

    # convert header info to floats
    i = 0
    dims = np.zeros([3])
    for d in dim_str.split(' '):
        dims[i] = d
        i = i + 1

    i = 0
    origin = np.zeros([3])
    for x in origin_str.split(' '):
        origin[i] = x
        i = i + 1

    dx = float(dx_str)

    # read in all sdf values
    sdf_grid = np.zeros(dims.astype('int64'))
    i = 0
    j = 0
    k = 0
    for line in f:
        if k < dims[2]:
            sdf_grid[i, j, k] = float(line)
            i = i + 1
            if i == dims[0]:
                i = 0
                j = j + 1
            if j == dims[1]:
                j = 0
                k = k + 1
    sdf_ = SDF(dims, origin, dx, sdf_grid)
    return sdf_


def render_sdf(obj_, object_name_):
    plt.figure()
    # ax = h.add_subplot(111, projection='3d')

    # surface_points = np.where(np.abs(sdf.sdf_values) < thresh)
    # surface_points = np.array(surface_points)
    # surface_points = surface_points[:, np.random.choice(surface_points[0].size, 3000, replace=True)]
    # # from IPython import embed; embed()
    surface_points = obj_.sdf.surface_points()[0]
    surface_points = np.array(surface_points)
    ind = np.random.choice(np.arange(len(surface_points)), 1000)
    x = surface_points[ind, 0]
    y = surface_points[ind, 1]
    z = surface_points[ind, 2]

    ax = plt.gca(projection=Axes3D.name)
    ax.scatter(x, y, z, '.', s=np.ones_like(x) * 0.3, c='b')
    ax.set_xlim3d(0, obj_.sdf.dims_[0])
    ax.set_ylim3d(0, obj_.sdf.dims_[1])
    ax.set_zlim3d(0, obj_.sdf.dims_[2])
    plt.title(object_name_)
    plt.show()


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    file_dir = home_dir + "/dataset/ycb_meshes_google/objects"

    if len(sys.argv) > 1:
        sdf_filename = sys.argv[1]
    else:
        sdf_filename = DEF_SDF_FILE
    # sdf_file = read_sdf(sdf_filename)
    # render_sdf(sdf_file, 1)

    obj_files = get_file_name(file_dir)
    obj_files = obj_files[1:4]
    for items in obj_files:
        object_name = items[len(home_dir) + 35:]

        sf = SdfFile(items + "/google_512k/nontextured.sdf")
        of = ObjFile(items + "/google_512k/nontextured.obj")
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)
        render_sdf(obj, object_name)
