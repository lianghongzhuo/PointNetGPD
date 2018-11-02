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
Example demonstrating the workflow for grasp planning on a phyiscal robot using point cloud registration.
Indexes the most robust grasp from the database and transforms it into the robot frame of reference. 

Authors
-------
Alan Li and Jeff Mahler

YAML Configuration File Parameters
----------------------------------
database : str
    full path to a Dex-Net HDF5 database
dataset : str
    name of the dataset containing the object instance to grasp
gripper : str
    name of the gripper to use
metric : str
    name of the grasp robustness metric to use
object : str
    name of the object to use (in practice instance recognition is necessary to determine the object instance from images)
"""
import os
from autolab_core import RigidTransform
from dexnet import DexNet
from dexnet.grasping import RobotGripper

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Example to demonstrate how to use grasps from a Dex-Net HDF5 database to execute grasps on a physical robot using point cloud registration')
    arser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/examples/execute_grasp_registration.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # parse config
    config = YamlConfig(config_filename)
    database_name = config['database']
    dataset_name = config['dataset']
    gripper_name = config['gripper']
    metric_name = config['metric']
    object_name = config['object']

    # fake transformation from the camera frame of reference to the robot frame of reference
    # in practice this could be computed by registering the camera to the robot base frame with a chessboard
    T_camera_robot = RigidTransform(rotation=RigidTransform.random_rotation(),
                                    translation=RigidTransform.random_translation(),
                                    from_frame='camera', to_frame='robot')

    # fake transformation from a known 3D object model frame of reference to the camera frame of reference
    # in practice this could be computed using point cloud registration
    T_obj_camera = RigidTransform(rotation=RigidTransform.random_rotation(),
                                  translation=RigidTransform.random_translation(),
                                  from_frame='obj', to_frame='camera')
    
    # load gripper
    gripper = RobotGripper.load(gripper_name)

    # open Dex-Net API
    dexnet_handle = DexNet()
    dexnet_handle.open_database(database_path)
    dexnet_handle.open_dataset(dataset_name)

    # read the most robust grasp
    sorted_grasps = dexnet_handle.dataset.sorted_grasps(object_name, metric_name, gripper=gripper_name)
    most_robust_grasp = sorted_grasps[0][0] # Note: in general this step will require collision checking

    # transform into the robot reference frame for control
    T_gripper_obj = most_robust_grasp.gripper_pose(gripper)
    T_gripper_robot = T_camera_robot * T_obj_camera * T_gripper_obj

    # control the robot to move along a linear path to the desired pose, close the jaws, and lift!
    
    
