"""
Script to generate a urdf for a mesh with a convex decomposition to preserve the geometry
Author: Jeff
"""
import argparse
import glob
import IPython
import logging
import numpy as np
import os
from subprocess import Popen
import sys

import xml.etree.cElementTree as et

from autolab_core import YamlConfig
from meshpy import Mesh3D, ObjFile, UrdfWriter

OBJ_EXT = '.obj'

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # read args
    parser = argparse.ArgumentParser(description='Convert a mesh to a URDF')
    parser.add_argument('mesh_filename', type=str, help='OBJ filename of the mesh to convert')
    parser.add_argument('output_dir', type=str, help='directory to store output urdf in')
    parser.add_argument('--config', type=str, default='cfg/tools/convex_decomposition.yaml',
                        help='config file for urdf conversion')
    args = parser.parse_args()

    # open config
    config_filename = args.config
    config = YamlConfig(config_filename)

    # check valid mesh filename
    mesh_filename = args.mesh_filename
    mesh_root, mesh_ext = os.path.splitext(mesh_filename)
    if mesh_ext.lower() != OBJ_EXT:
        logging.error('Extension %s not supported' %(mesh_ext))
        exit(0)

    # open mesh
    of = ObjFile(mesh_filename)
    mesh = of.read()
    mesh.density = config['object_density']

    # create output dir for urdf
    output_dir = args.output_dir
    writer = UrdfWriter(output_dir)
    writer.write(mesh)
