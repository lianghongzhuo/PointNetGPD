# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Visualizes point clouds from a generated GQ-CNN training dataset, optionally filtering by the grasp robustness metrics.

Author
------
Jeff Mahler
"""
import argparse
import copy
import IPython
import logging
import numpy as np
import os
import sys

import autolab_core.utils as utils
from autolab_core import Point, YamlConfig
from perception import BinaryImage, ColorImage, DepthImage, GdImage, GrayscaleImage, RgbdImage, RenderMode

from gqcnn import Grasp2D
from gqcnn import Visualizer as vis2d

from dexnet.learning import TensorDataset

def visualize_tensor_dataset(dataset, config):
    """
    Visualizes a Tensor dataset.

    Parameters
    ----------
    dataset : :obj:`TensorDataset`
        dataset to visualize
    config : :obj:`autolab_core.YamlConfig`
        parameters for visualization

    Notes
    -----
    Required parameters of config are specified in Other Parameters

    Other Parameters
    ----------------
    field_name : str
        name of the field in the TensorDataset to visualize (defaults to depth_ims_tf_table, which is a single view point cloud of the object on a table)
    field_type : str
        type of image that the field name correspondes to (defaults to depth, can also be `segmask` if using the field `object_masks`)

    print_fields : :obj:`list` of str
        names of additiona fields to print to the command line
    filter : :obj:`dict` mapping str to :obj:`dict` 
        contraints that all displayed datapoints must satisfy (supports any univariate field name as a key and numeric thresholds)

    gripper_width_px : float
        width of the gripper to plot in pixels
    font_size : int
        size of font on the rendered images
    """
    # shuffle the tensor indices
    indices = dataset.datapoint_indices
    np.random.shuffle(indices)

    # read config
    field_name = config['field_name']
    field_type = config['field_type']
    font_size = config['font_size']
    print_fields = config['print_fields']
    gripper_width_px = config['gripper_width_px']

    num = 0
    for i, ind in enumerate(indices):
        datapoint = dataset[ind]
        data = datapoint[field_name]
        if field_type == RenderMode.SEGMASK:
            image = BinaryImage(data)
        elif field_type == RenderMode.DEPTH:
            image = DepthImage(data)
        else:
            raise ValueError('Field type %s not supported!' %(field_type))

        skip_datapoint = False
        for f, filter_cfg in config['filter'].iteritems():
            data = datapoint[f]
            if 'greater_than' in filter_cfg.keys() and data < filter_cfg['greater_than']:
                skip_datapoint = True
                break
            elif 'less_than' in filter_cfg.keys() and data > filter_cfg['less_than']:
                skip_datapoint = True
                break
        if skip_datapoint:
            continue

        logging.info('DATAPOINT %d' %(num))
        for f in print_fields:
            data = datapoint[f]
            logging.info('Field %s:' %(f))
            print(data)

        grasp_2d = Grasp2D(Point(image.center), 0, datapoint['hand_poses'][2])

        vis2d.figure()
        if field_type == RenderMode.RGBD:
            vis2d.subplot(1,2,1)
            vis2d.imshow(image.color)
            vis2d.grasp(grasp_2d, width=gripper_width_px)
            vis2d.subplot(1,2,2)
            vis2d.imshow(image.depth)
            vis2d.grasp(grasp_2d, width=gripper_width_px)
        elif field_type == RenderMode.GD:
            vis2d.subplot(1,2,1)
            vis2d.imshow(image.gray)
            vis2d.grasp(grasp_2d, width=gripper_width_px)
            vis2d.subplot(1,2,2)
            vis2d.imshow(image.depth)
            vis2d.grasp(grasp_2d, width=gripper_width_px)
        else:
            vis2d.imshow(image)
            vis2d.grasp(grasp_2d, width=gripper_width_px)
        vis2d.title('Datapoint %d: %s' %(ind, field_type))
        vis2d.show()
            
        num += 1

if __name__ == '__main__':
    # parse args
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, default=None, help='Path to the dataset to visualize')
    parser.add_argument('--config_filename', type=str, default=None, help='Yaml filename containing configuration parameters for the visualization')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    config_filename = args.config_filename

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/visualize_gqcnn_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # read config
    config = YamlConfig(config_filename)

    # open tensor dataset
    dataset = TensorDataset.open(dataset_path)

    # visualize a tensor dataset
    visualize_tensor_dataset(dataset, config)
