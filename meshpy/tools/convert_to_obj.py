"""
Script to convert a directory of 3D models to .OBJ wavefront format for use in meshpy using meshlabserver.
Author: Jeff Mahler
"""
import argparse
import logging
import os
import sys

import autolab_core.utils as utils

SUPPORTED_EXTENSIONS = ['.wrl', '.obj', '.off', '.ply', '.stl', '.3ds']

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Convert a directory of 3D models into .OBJ format using meshlab')
    parser.add_argument('input_dir', type=str, help='directory containing 3D model files to convert')
    parser.add_argument('--output_dir', type=str, default=None, help='directory to save .OBJ files to')
    args = parser.parse_args()
    data_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = data_dir

    # get model filenames
    model_filenames = []
    for ext in SUPPORTED_EXTENSIONS:
        model_filenames.extend(utils.filenames(data_dir, tag=ext))
    model_file_roots = []
    for model_filename in model_filenames:
        root, _ = os.path.splitext(model_filename)
        model_file_roots.append(root)
    # create obj filenames
    obj_filenames = [f + '.obj' for f in model_file_roots]
    obj_filenames = [f.replace(data_dir, output_dir) for f in obj_filenames]
    num_files = len(obj_filenames)

    # convert using meshlab server
    i = 0
    for obj_filename, model_filename in zip(obj_filenames, model_filenames):
        logging.info('Converting %s (%d of %d)' %(model_filename, i+1, num_files))
        
        # call meshlabserver
        meshlabserver_cmd = 'meshlabserver -i %s -o %s' %(model_filename, obj_filename)
        os.system(meshlabserver_cmd)
        
        
        

