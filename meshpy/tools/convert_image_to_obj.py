"""
Script to convert a directory of 3D models to .OBJ wavefront format for use in meshpy using meshlabserver.
Author: Jeff Mahler
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import autolab_core.utils as utils
from perception import BinaryImage
from meshpy import ImageToMeshConverter, ObjFile
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis

if __name__ == '__main__':
        # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Convert an image into an extruded 3D mesh model')
    parser.add_argument('input_image', type=str, help='path to image to convert')
    parser.add_argument('--extrusion', type=float, default=1000, help='amount to extrude')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='scale factor to apply to the mesh')
    parser.add_argument('--output_filename', type=str, default=None, help='output obj filename')

    args = parser.parse_args()
    image_filename = args.input_image
    extrusion = args.extrusion
    scale_factor = args.scale_factor
    output_filename = args.output_filename

    # read the image
    binary_im = BinaryImage.open(image_filename)
    sdf = binary_im.to_sdf()
    #plt.figure()
    #plt.imshow(sdf)
    #plt.show()

    # convert to a mesh
    mesh = ImageToMeshConverter.binary_image_to_mesh(binary_im, extrusion=extrusion, scale_factor=scale_factor)
    vis.figure()
    vis.mesh(mesh)
    vis.show()

    # optionally save
    if output_filename is not None:
        file_root, file_ext = os.path.splitext(output_filename)
        binary_im.save(file_root+'.jpg')
        ObjFile(file_root+'.obj').write(mesh)
        np.savetxt(file_root+'.csv', sdf, delimiter=',',
                   header='%d %d'%(sdf.shape[0], sdf.shape[1]))
