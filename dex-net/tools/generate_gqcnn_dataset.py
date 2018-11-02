# -*- coding: utf-8 -*-
# """
# Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational,
# research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
# hereby granted, provided that the above copyright notice, this paragraph and the following two
# paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
# Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
# 7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
# THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# """
# """
# Generates datasets of synthetic point clouds, grasps, and grasp robustness metrics from a Dex-Net HDF5 database for GQ-CNN training.
#
# Author
# ------
# Jeff Mahler
#
# YAML Configuration File Parameters
# ----------------------------------
# database_name : str
#     full path to a Dex-Net HDF5 database
# target_object_keys : :obj:`OrderedDict`
#     dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
# env_rv_params : :obj:`OrderedDict`
#     parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
# gripper_name : str
#     name of the gripper to use
# """
import argparse
import collections
import pickle as pkl
import gc
import IPython
import json
import logging
import numpy as np
import os
import random
import shutil
import sys
import time

from autolab_core import Point, RigidTransform, YamlConfig
import autolab_core.utils as utils
from gqcnn import Grasp2D
from gqcnn import Visualizer as vis2d
from meshpy import ObjFile, RenderMode, SceneObject, UniformPlanarWorksurfaceImageRandomVariable
from perception import CameraIntrinsics, BinaryImage, DepthImage

from dexnet.constants import READ_ONLY_ACCESS
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.learning import TensorDataset

try:
    from dexnet.visualization import DexNetVisualizer3D as vis
except:
    logging.warning('Failed to import DexNetVisualizer3D, visualization methods will be unavailable')

logging.root.name = 'dex-net'

# seed for deterministic behavior when debugging
SEED = 197561

# name of the grasp cache file
CACHE_FILENAME = 'grasp_cache.pkl'


class GraspInfo(object):
    """ Struct to hold precomputed grasp attributes.
    For speeding up dataset generation.
    """

    def __init__(self, grasp, collision_free, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.phi = phi


def generate_gqcnn_dataset(dataset_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           config):
    """
    Generates a GQ-CNN TensorDataset for training models with new grippers, quality metrics, objects, and cameras.

    Parameters
    ----------
    dataset_path : str
        path to save the dataset to
    database : :obj:`Hdf5Database`
        Dex-Net database containing the 3D meshes, grasps, and grasp metrics
    target_object_keys : :obj:`OrderedDict`
        dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
    env_rv_params : :obj:`OrderedDict`
        parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
    gripper_name : str
        name of the gripper to use
    config : :obj:`autolab_core.YamlConfig`
        other parameters for dataset generation

    Notes
    -----
    Required parameters of config are specified in Other Parameters

    Other Parameters
    ----------------    
    images_per_stable_pose : int
        number of object and camera poses to sample for each stable pose
    stable_pose_min_p : float
        minimum probability of occurrence for a stable pose to be used in data generation (used to prune bad stable poses
    
    gqcnn/crop_width : int
        width, in pixels, of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/crop_height : int
        height, in pixels,  of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/final_width : int
        width, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)
    gqcnn/final_height : int
        height, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)

    table_alignment/max_approach_table_angle : float
        max angle between the grasp axis and the table normal when the grasp approach is maximally aligned with the table normal
    table_alignment/max_approach_offset : float
        max deviation from perpendicular approach direction to use in grasp collision checking
    table_alignment/num_approach_offset_samples : int
        number of approach samples to use in collision checking

    collision_checking/table_offset : float
        max allowable interpenetration between the gripper and table to be considered collision free
    collision_checking/table_mesh_filename : str
        path to a table mesh for collision checking (default data/meshes/table.obj)
    collision_checking/approach_dist : float
        distance, in meters, between the approach pose and final grasp pose along the grasp axis
    collision_checking/delta_approach : float
        amount, in meters, to discretize the straight-line path from the gripper approach pose to the final grasp pose

    tensors/datapoints_per_file : int
        number of datapoints to store in each unique tensor file on disk
    tensors/fields : :obj:`dict`
        dictionary mapping field names to dictionaries specifying the data type, height, width, and number of channels for each tensor

    debug : bool
        True (or 1) if the random seed should be set to enforce deterministic behavior, False (0) otherwise
    vis/candidate_grasps : bool
        True (or 1) if the collision free candidate grasps should be displayed in 3D (for debugging)
    vis/rendered_images : bool
        True (or 1) if the rendered images for each stable pose should be displayed (for debugging)
    vis/grasp_images : bool
        True (or 1) if the transformed grasp images should be displayed (for debugging)
    """
    # read data gen params
    output_dir = dataset_path
    gripper = RobotGripper.load(gripper_name)
    image_samples_per_stable_pose = config['images_per_stable_pose']
    stable_pose_min_p = config['stable_pose_min_p']

    # read gqcnn params
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2

    # open database
    dataset_names = target_object_keys.keys()
    datasets = [database.dataset(dn) for dn in dataset_names]

    # set target objects
    for dataset in datasets:
        if target_object_keys[dataset.name] == 'all':
            target_object_keys[dataset.name] = dataset.object_keys

    # setup grasp params
    table_alignment_params = config['table_alignment']
    min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
    num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

    phi_offsets = []
    if max_grasp_approach_offset == min_grasp_approach_offset:
        phi_inc = 1
    elif num_grasp_approach_samples == 1:
        phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
    else:
        phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)

    phi = min_grasp_approach_offset
    while phi <= max_grasp_approach_offset:
        phi_offsets.append(phi)
        phi += phi_inc

    # setup collision checking
    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    if not os.path.isabs(table_mesh_filename):
        table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', table_mesh_filename)
    table_mesh = ObjFile(table_mesh_filename).read()

    # set tensor dataset config
    tensor_config = config['tensors']
    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    tensor_config['fields']['obj_masks']['height'] = im_final_height
    tensor_config['fields']['obj_masks']['width'] = im_final_width

    # add available metrics (assuming same are computed for all objects)
    metric_names = []
    dataset = datasets[0]
    obj_keys = dataset.object_keys
    if len(obj_keys) == 0:
        raise ValueError('No valid objects in dataset %s' % (dataset.name))

    obj = dataset[obj_keys[0]]
    grasps = dataset.grasps(obj.key, gripper=gripper.name)
    grasp_metrics = dataset.grasp_metrics(obj.key, grasps, gripper=gripper.name)
    metric_names = grasp_metrics[grasp_metrics.keys()[0]].keys()
    for metric_name in metric_names:
        tensor_config['fields'][metric_name] = {}
        tensor_config['fields'][metric_name]['dtype'] = 'float32'

    # init tensor dataset
    tensor_dataset = TensorDataset(output_dir, tensor_config)
    tensor_datapoint = tensor_dataset.datapoint_template

    # setup log file
    experiment_log_filename = os.path.join(output_dir, 'dataset_generation.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    hdlr = logging.FileHandler(experiment_log_filename)
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr)
    root_logger = logging.getLogger()

    # copy config
    out_config_filename = os.path.join(output_dir, 'dataset_generation.json')
    ordered_dict_config = collections.OrderedDict()
    for key in config.keys():
        ordered_dict_config[key] = config[key]
    with open(out_config_filename, 'w') as outfile:
        json.dump(ordered_dict_config, outfile)

    # 1. Precompute the set of valid grasps for each stable pose:
    #    i) Perpendicular to the table
    #   ii) Collision-free along the approach direction

    # load grasps if they already exist
    grasp_cache_filename = os.path.join(output_dir, CACHE_FILENAME)
    if os.path.exists(grasp_cache_filename):
        logging.info('Loading grasp candidates from file')
        candidate_grasps_dict = pkl.load(open(grasp_cache_filename, 'rb'))
    # otherwise re-compute by reading from the database and enforcing constraints
    else:
        # create grasps dict
        candidate_grasps_dict = {}

        # loop through datasets and objects
        for dataset in datasets:
            logging.info('Reading dataset %s' % (dataset.name))
            for obj in dataset:
                if obj.key not in target_object_keys[dataset.name]:
                    continue

                # init candidate grasp storage
                candidate_grasps_dict[obj.key] = {}

                # setup collision checker
                collision_checker = GraspCollisionChecker(gripper)
                collision_checker.set_graspable_object(obj)

                # read in the stable poses of the mesh
                stable_poses = dataset.stable_poses(obj.key)
                for i, stable_pose in enumerate(stable_poses):
                    # render images if stable pose is valid
                    if stable_pose.p > stable_pose_min_p:
                        candidate_grasps_dict[obj.key][stable_pose.id] = []

                        # setup table in collision checker
                        T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                        T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj',
                                                                                                          'table')
                        T_table_obj = T_obj_table.inverse()
                        collision_checker.set_table(table_mesh_filename, T_table_obj)

                        # read grasp and metrics
                        grasps = dataset.grasps(obj.key, gripper=gripper.name)
                        logging.info(
                            'Aligning %d grasps for object %s in stable %s' % (len(grasps), obj.key, stable_pose.id))

                        # align grasps with the table
                        aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]

                        # check grasp validity
                        logging.info('Checking collisions for %d grasps for object %s in stable %s' % (
                        len(grasps), obj.key, stable_pose.id))
                        for aligned_grasp in aligned_grasps:
                            # check angle with table plane and skip unaligned grasps
                            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                            perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                            if not perpendicular_table:
                                continue

                            # check whether any valid approach directions are collision free
                            collision_free = False
                            for phi_offset in phi_offsets:
                                rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                                collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist,
                                                                                     delta_approach)
                                if not collides:
                                    collision_free = True
                                    break

                            # store if aligned to table
                            candidate_grasps_dict[obj.key][stable_pose.id].append(
                                GraspInfo(aligned_grasp, collision_free))

                            # visualize if specified
                            if collision_free and config['vis']['candidate_grasps']:
                                logging.info('Grasp %d' % (aligned_grasp.id))
                                vis.figure()
                                vis.gripper_on_object(gripper, aligned_grasp, obj, stable_pose.T_obj_world)
                                vis.show()

        # save to file
        logging.info('Saving to file')
        pkl.dump(candidate_grasps_dict, open(grasp_cache_filename, 'wb'))

    # 2. Render a dataset of images and associate the gripper pose with image coordinates for each grasp in the
    # Dex-Net database

    # setup variables
    obj_category_map = {}
    pose_category_map = {}

    cur_pose_label = 0
    cur_obj_label = 0
    cur_image_label = 0

    # render images for each stable pose of each object in the dataset
    render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]
    for dataset in datasets:
        logging.info('Generating data for dataset %s' % (dataset.name))

        # iterate through all object keys
        object_keys = dataset.object_keys
        for obj_key in object_keys:
            obj = dataset[obj_key]
            if obj.key not in target_object_keys[dataset.name]:
                continue

            # read in the stable poses of the mesh
            stable_poses = dataset.stable_poses(obj.key)
            for i, stable_pose in enumerate(stable_poses):

                # render images if stable pose is valid
                if stable_pose.p > stable_pose_min_p:
                    # log progress
                    logging.info('Rendering images for object %s in %s' % (obj.key, stable_pose.id))

                    # add to category maps
                    if obj.key not in obj_category_map.keys():
                        obj_category_map[obj.key] = cur_obj_label
                    pose_category_map['%s_%s' % (obj.key, stable_pose.id)] = cur_pose_label

                    # read in candidate grasps and metrics
                    candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                    candidate_grasps = [g.grasp for g in candidate_grasp_info]
                    grasp_metrics = dataset.grasp_metrics(obj.key, candidate_grasps, gripper=gripper.name)

                    # compute object pose relative to the table
                    T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                    T_obj_stp = obj.mesh.get_T_surface_obj(T_obj_stp)

                    # sample images from random variable
                    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
                    scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}
                    urv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                                      render_modes,
                                                                      'camera',
                                                                      env_rv_params,
                                                                      stable_pose=stable_pose,
                                                                      scene_objs=scene_objs)

                    render_start = time.time()
                    render_samples = urv.rvs(size=image_samples_per_stable_pose)
                    render_stop = time.time()
                    logging.info('Rendering images took %.3f sec' % (render_stop - render_start))

                    # visualize
                    if config['vis']['rendered_images']:
                        d = int(np.ceil(np.sqrt(image_samples_per_stable_pose)))

                        # binary
                        vis2d.figure()
                        for j, render_sample in enumerate(render_samples):
                            vis2d.subplot(d, d, j + 1)
                            vis2d.imshow(render_sample.renders[RenderMode.SEGMASK].image)

                        # depth table
                        vis2d.figure()
                        for j, render_sample in enumerate(render_samples):
                            vis2d.subplot(d, d, j + 1)
                            vis2d.imshow(render_sample.renders[RenderMode.DEPTH_SCENE].image)
                        vis2d.show()

                    # tally total amount of data
                    num_grasps = len(candidate_grasps)
                    num_images = image_samples_per_stable_pose
                    num_save = num_images * num_grasps
                    logging.info('Saving %d datapoints' % (num_save))

                    # for each candidate grasp on the object compute the projection
                    # of the grasp into image space
                    for render_sample in render_samples:
                        # read images
                        binary_im = render_sample.renders[RenderMode.SEGMASK].image
                        depth_im_table = render_sample.renders[RenderMode.DEPTH_SCENE].image
                        # read camera params
                        T_stp_camera = render_sample.camera.object_to_camera_pose
                        shifted_camera_intr = render_sample.camera.camera_intr

                        # read pixel offsets
                        cx = depth_im_table.center[1]
                        cy = depth_im_table.center[0]

                        # compute intrinsics for virtual camera of the final
                        # cropped and rescaled images
                        camera_intr_scale = float(im_final_height) / float(im_crop_height)
                        cropped_camera_intr = shifted_camera_intr.crop(im_crop_height, im_crop_width, cy, cx)
                        final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)

                        # create a thumbnail for each grasp
                        for grasp_info in candidate_grasp_info:
                            # read info
                            grasp = grasp_info.grasp
                            collision_free = grasp_info.collision_free

                            # get the gripper pose
                            T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)
                            grasp_2d = grasp.project_camera(T_obj_camera, shifted_camera_intr)

                            # center images on the grasp, rotate to image x axis
                            dx = cx - grasp_2d.center.x
                            dy = cy - grasp_2d.center.y
                            translation = np.array([dy, dx])

                            binary_im_tf = binary_im.transform(translation, grasp_2d.angle)
                            depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)

                            # crop to image size
                            binary_im_tf = binary_im_tf.crop(im_crop_height, im_crop_width)
                            depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)

                            # resize to image size
                            binary_im_tf = binary_im_tf.resize((im_final_height, im_final_width), interp='nearest')
                            depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))

                            # visualize the transformed images
                            if config['vis']['grasp_images']:
                                grasp_center = Point(depth_im_tf_table.center,
                                                     frame=final_camera_intr.frame)
                                tf_grasp_2d = Grasp2D(grasp_center, 0,
                                                      grasp_2d.depth,
                                                      width=gripper.max_width,
                                                      camera_intr=final_camera_intr)

                                # plot 2D grasp image
                                vis2d.figure()
                                vis2d.subplot(2, 2, 1)
                                vis2d.imshow(binary_im)
                                vis2d.grasp(grasp_2d)
                                vis2d.subplot(2, 2, 2)
                                vis2d.imshow(depth_im_table)
                                vis2d.grasp(grasp_2d)
                                vis2d.subplot(2, 2, 3)
                                vis2d.imshow(binary_im_tf)
                                vis2d.grasp(tf_grasp_2d)
                                vis2d.subplot(2, 2, 4)
                                vis2d.imshow(depth_im_tf_table)
                                vis2d.grasp(tf_grasp_2d)
                                vis2d.title('Coll Free? %d' % (grasp_info.collision_free))
                                vis2d.show()

                                # plot 3D visualization
                                vis.figure()
                                T_obj_world = vis.mesh_stable_pose(obj.mesh, stable_pose.T_obj_world, style='surface',
                                                                   dim=0.5)
                                vis.gripper(gripper, grasp, T_obj_world, color=(0.3, 0.3, 0.3))
                                vis.show()

                            # form hand pose array
                            hand_pose = np.r_[grasp_2d.center.y,
                                              grasp_2d.center.x,
                                              grasp_2d.depth,
                                              grasp_2d.angle,
                                              grasp_2d.center.y - shifted_camera_intr.cy,
                                              grasp_2d.center.x - shifted_camera_intr.cx,
                                              grasp_2d.width_px]

                            # store to data buffers
                            tensor_datapoint['depth_ims_tf_table'] = depth_im_tf_table.raw_data
                            tensor_datapoint['obj_masks'] = binary_im_tf.raw_data
                            tensor_datapoint['hand_poses'] = hand_pose
                            tensor_datapoint['collision_free'] = collision_free
                            tensor_datapoint['obj_labels'] = cur_obj_label
                            tensor_datapoint['pose_labels'] = cur_pose_label
                            tensor_datapoint['image_labels'] = cur_image_label

                            for metric_name, metric_val in grasp_metrics[grasp.id].iteritems():
                                coll_free_metric = (1 * collision_free) * metric_val
                                tensor_datapoint[metric_name] = coll_free_metric
                            tensor_dataset.add(tensor_datapoint)

                        # update image label
                        cur_image_label += 1

                    # update pose label
                    cur_pose_label += 1

                    # force clean up
                    gc.collect()

            # update object label
            cur_obj_label += 1

            # force clean up
            gc.collect()

    # save last file
    tensor_dataset.flush()

    # save category mappings
    obj_cat_filename = os.path.join(output_dir, 'object_category_map.json')
    json.dump(obj_category_map, open(obj_cat_filename, 'w'))
    pose_cat_filename = os.path.join(output_dir, 'pose_category_map.json')
    json.dump(pose_category_map, open(pose_cat_filename, 'w'))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(
        description='Create a GQ-CNN training dataset from a dataset of 3D object models and grasps in a Dex-Net database')
    parser.add_argument('dataset_path', type=str, default=None, help='name of folder to save the training dataset in')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    config_filename = args.config_filename

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/generate_gqcnn_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # parse config
    config = YamlConfig(config_filename)

    # set seed
    debug = config['debug']
    if debug:
        random.seed(SEED)
        np.random.seed(SEED)

    # open database
    database = Hdf5Database(config['database_name'],
                            access_level=READ_ONLY_ACCESS)

    # read params
    target_object_keys = config['target_objects']
    env_rv_params = config['env_rv_params']
    gripper_name = config['gripper']

    # generate the dataset
    generate_gqcnn_dataset(dataset_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           config)
