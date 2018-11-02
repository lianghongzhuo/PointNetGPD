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
A bunch of classes for converting hdf5 groups & datasets to common object types
Author: Jeff Mahler
"""
import datetime as dt
import h5py
import IPython
import logging
import numpy as np

import meshpy.mesh as mesh
import meshpy.sdf as sdf
import meshpy.stable_pose as stp

from autolab_core import RigidTransform
from perception import BinaryImage, ColorImage, DepthImage, ObjectRender, RenderMode
import perception as f

from dexnet.database.keys import *
from dexnet.grasping import ParallelJawPtGrasp3D

class Hdf5ObjectFactory(object):
    """ Functions for reading and writing new objects from HDF5 fields. Should not be called directly. """

    @staticmethod
    def sdf_3d(data):
        """ Converts HDF5 data provided in dictionary data to an SDF object """
        sdf_data = np.array(data[SDF_DATA_KEY])
        origin = np.array(data.attrs[SDF_ORIGIN_KEY])
        resolution = data.attrs[SDF_RES_KEY]
        
        return sdf.Sdf3D(sdf_data, origin, resolution)

    @staticmethod
    def write_sdf_3d(sdf, data):
        """ Writes sdf object to HDF5 data provided in data """
        data.create_dataset(SDF_DATA_KEY, data=sdf.data)
        data.attrs.create(SDF_ORIGIN_KEY, sdf.origin)
        data.attrs.create(SDF_RES_KEY, sdf.resolution)
        
    @staticmethod
    def mesh_3d(data):
        """ Converts HDF5 data provided in dictionary data to a mesh object """
        vertices = np.array(data[MESH_VERTICES_KEY])
        triangles = np.array(data[MESH_TRIANGLES_KEY])

        normals = None
        if MESH_NORMALS_KEY in list(data.keys()):
            normals = np.array(data[MESH_NORMALS_KEY])
        return mesh.Mesh3D(vertices, triangles, normals=normals)

    @staticmethod
    def write_mesh_3d(mesh, data):
        """ Writes mesh object to HDF5 data provided in data """
        data.create_dataset(MESH_VERTICES_KEY, data=mesh.vertices)
        data.create_dataset(MESH_TRIANGLES_KEY, data=mesh.triangles)
        if mesh.normals is not None:
            data.create_dataset(MESH_NORMALS_KEY, data=mesh.normals)

    @staticmethod
    def stable_poses(data):
        """ Read out a list of stable pose objects """
        num_stable_poses = data.attrs[NUM_STP_KEY]
        stable_poses = []
        for i in range(num_stable_poses):
            stp_key = POSE_KEY + '_' + str(i) 
            p = data[stp_key].attrs[STABLE_POSE_PROB_KEY]
            r = data[stp_key].attrs[STABLE_POSE_ROT_KEY]
            try:
                x0 = data[stp_key].attrs[STABLE_POSE_PT_KEY]
            except:
                x0 = np.zeros(3)
            stable_poses.append(stp.StablePose(p, r, x0, stp_id=stp_key))
        return stable_poses

    @staticmethod
    def stable_pose(data, stable_pose_id):
        """ Read out a stable pose object """
        p = data[stable_pose_id].attrs[STABLE_POSE_PROB_KEY]
        r = data[stable_pose_id].attrs[STABLE_POSE_ROT_KEY]
        try:
            x0 = data[stable_pose_id].attrs[STABLE_POSE_PT_KEY]
        except:
            x0 = np.zeros(3)
        return stp.StablePose(p, r, x0, stp_id=stable_pose_id)

    @staticmethod
    def write_stable_poses(stable_poses, data, force_overwrite=False):
        """ Writes stable poses to HDF5 data provided in data """
        num_stable_poses = len(stable_poses)
        data.attrs.create(NUM_STP_KEY, num_stable_poses)
        for i, stable_pose in enumerate(stable_poses):
            stp_key = POSE_KEY + '_' + str(i)
            if stp_key not in list(data.keys()):
                data.create_group(stp_key)
                data[stp_key].attrs.create(STABLE_POSE_PROB_KEY, stable_pose.p)
                data[stp_key].attrs.create(STABLE_POSE_ROT_KEY, stable_pose.r)
                data[stp_key].attrs.create(STABLE_POSE_PT_KEY, stable_pose.x0)
                data[stp_key].create_group(RENDERED_IMAGES_KEY)
            elif force_overwrite:
                data[stp_key].attrs[STABLE_POSE_PROB_KEY] = stable_pose.p
                data[stp_key].attrs[STABLE_POSE_ROT_KEY] = stable_pose.r
                data[stp_key].attrs[STABLE_POSE_PT_KEY] = stable_pose.x0
            else:
                logging.warning('Stable %s already exists and overwrite was not requested. Aborting write request' %(stp_key))
                return None

    @staticmethod
    def grasps(data):
        """ Return a list of grasp objects from the data provided in the HDF5 dictionary """
        # need to read in a bunch of grasps but also need to know what kind of grasp it is
        grasps = []
        num_grasps = data.attrs[NUM_GRASPS_KEY]
        for i in range(num_grasps):
            # get the grasp data y'all
            grasp_key = GRASP_KEY + '_' + str(i)
            if grasp_key in list(data.keys()):
                grasp_id =      data[grasp_key].attrs[GRASP_ID_KEY]            
                grasp_type =    data[grasp_key].attrs[GRASP_TYPE_KEY]
                configuration = data[grasp_key].attrs[GRASP_CONFIGURATION_KEY]
                frame =         data[grasp_key].attrs[GRASP_RF_KEY]            
                
                # create object based on type
                g = None
                if grasp_type == 'ParallelJawPtGrasp3D':
                    g = ParallelJawPtGrasp3D(configuration=configuration, frame=frame, grasp_id=grasp_id)
                grasps.append(g)
            else:
                logging.debug('Grasp %s is corrupt. Skipping' %(grasp_key))

        return grasps

    @staticmethod
    def write_grasps(grasps, data, force_overwrite=False):
        """ Writes grasps to HDF5 data provided in data """
        num_grasps = data.attrs[NUM_GRASPS_KEY]
        num_new_grasps = len(grasps)

        # get timestamp for pruning old grasps
        dt_now = dt.datetime.now()
        creation_stamp = '%s-%s-%s-%sh-%sm-%ss' %(dt_now.month, dt_now.day, dt_now.year, dt_now.hour, dt_now.minute, dt_now.second) 

        # add each grasp
        for i, grasp in enumerate(grasps):
            grasp_id = grasp.id
            if grasp_id is None:
                grasp_id = i+num_grasps
            grasp_key = GRASP_KEY + '_' + str(grasp_id)

            if grasp_key not in list(data.keys()):
                data.create_group(grasp_key)
                data[grasp_key].attrs.create(GRASP_ID_KEY, grasp_id)
                data[grasp_key].attrs.create(GRASP_TYPE_KEY, type(grasp).__name__)
                data[grasp_key].attrs.create(GRASP_CONFIGURATION_KEY, grasp.configuration)
                data[grasp_key].attrs.create(GRASP_RF_KEY, grasp.frame)
                data[grasp_key].create_group(GRASP_METRICS_KEY) 
            elif force_overwrite:
                data[grasp_key].attrs[GRASP_ID_KEY] = grasp_id
                data[grasp_key].attrs[GRASP_TYPE_KEY] = type(grasp).__name__
                data[grasp_key].attrs[GRASP_CONFIGURATION_KEY] = grasp.configuration
                data[grasp_key].attrs[GRASP_RF_KEY] = grasp.frame
            else:
                logging.warning('Grasp %d already exists and overwrite was not requested. Aborting write request' %(grasp_id))
                return None

        data.attrs[NUM_GRASPS_KEY] = num_grasps + num_new_grasps
        return creation_stamp

    @staticmethod
    def grasp_metrics(grasps, data):
        """ Returns a dictionary of the metrics for the given grasps """
        grasp_metrics = {}
        for grasp in grasps:
            grasp_id = grasp.id
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            grasp_metrics[grasp_id] = {}
            if grasp_key in list(data.keys()):
                grasp_metric_data = data[grasp_key][GRASP_METRICS_KEY]                
                for metric_name in list(grasp_metric_data.attrs.keys()):
                    grasp_metrics[grasp_id][metric_name] = grasp_metric_data.attrs[metric_name]
        return grasp_metrics

    @staticmethod
    def write_grasp_metrics(grasp_metric_dict, data, force_overwrite=False):
        """ Write grasp metrics to database """
        for grasp_id, metric_dict in grasp_metric_dict.items():
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            if grasp_key in list(data.keys()):
                grasp_metric_data = data[grasp_key][GRASP_METRICS_KEY]

                for metric_tag, metric in metric_dict.items():
                    if metric_tag not in list(grasp_metric_data.attrs.keys()):
                        grasp_metric_data.attrs.create(metric_tag, metric)
                    elif force_overwrite:
                        grasp_metric_data.attrs[metric_tag] = metric
                    else:
                        logging.warning('Metric %s already exists for grasp %s and overwrite was not requested. Aborting write request' %(metric_tag, grasp_id))
                        return False
        return True

    @staticmethod
    def rendered_images(data, render_mode=RenderMode.SEGMASK):
        rendered_images = []
        num_images = data.attrs[NUM_IMAGES_KEY]
        
        for i in range(num_images):
            # get the image data y'all
            image_key = IMAGE_KEY + '_' + str(i)
            image_data = data[image_key]
            image_arr = np.array(image_data[IMAGE_DATA_KEY])
            frame = image_data.attrs[IMAGE_FRAME_KEY]
            if render_mode == RenderMode.SEGMASK:
                image = BinaryImage(image_arr, frame)
            elif render_mode == RenderMode.DEPTH:
                image = DepthImage(image_arr, frame)
            elif render_mode == RenderMode.SCALED_DEPTH:
                image = ColorImage(image_arr, frame)
            R_camera_table =  image_data.attrs[CAM_ROT_KEY]
            t_camera_table =  image_data.attrs[CAM_POS_KEY]
            frame          =  image_data.attrs[CAM_FRAME_KEY]
            T_camera_world = RigidTransform(R_camera_table, t_camera_table,
                                            from_frame=frame,
                                            to_frame='world')
            
            rendered_images.append(ObjectRender(image, T_camera_world))
        return rendered_images

    @staticmethod
    def write_rendered_images(rendered_images, data, force_overwrite=False):
        """ Write rendered images to database """
        num_images = 0
        if NUM_IMAGES_KEY in list(data.keys()):
            num_images = data.attrs[NUM_GRASPS_KEY]
        num_new_images = len(rendered_images)        

        for image_id, rendered_image in enumerate(rendered_images):
            if not isinstance(rendered_image, ObjectRender):
                raise ValueError('Must provide images of type ObjectRender')

            image_key = IMAGE_KEY + '_' + str(image_id)
            if image_key not in list(data.keys()):
                data.create_group(image_key)
                image_data = data[image_key]

                image_data.create_dataset(IMAGE_DATA_KEY, data=rendered_image.image.data)
                image_data.attrs.create(IMAGE_FRAME_KEY, rendered_image.image.frame)
                image_data.attrs.create(CAM_ROT_KEY, rendered_image.T_camera_world.rotation)
                image_data.attrs.create(CAM_POS_KEY, rendered_image.T_camera_world.translation)
                image_data.attrs.create(CAM_FRAME_KEY, rendered_image.T_camera_world.from_frame)
            elif force_overwrite:
                image_data[IMAGE_DATA_KEY] = rendered_image.image
                image_data.attrs[CAM_ROT_KEY] = rendered_image.T_camera_world.rotation
                image_data.attrs[CAM_POS_KEY] = rendered_image.T_camera_world.translation
                image_data.attrs[CAM_FRAME_KEY] = rendered_image.T_camera_world.from_frame
            else:
                logging.warning('Image %d already exists and overwrite was not requested. Aborting write request' %(image_id))
                return None

        if NUM_IMAGES_KEY in list(data.keys()):
            data.attrs[NUM_IMAGES_KEY] = num_images + num_new_images
        else:
            data.attrs.create(NUM_IMAGES_KEY, num_images + num_new_images)
            
    @staticmethod
    def connected_components(data):
        """ Returns a dict of all connected components in the object """
        if CONNECTED_COMPONENTS_KEY not in list(data.keys()):
            return None
        out = {}
        for key in data[CONNECTED_COMPONENTS_KEY]:
            out[key] = Hdf5ObjectFactory.mesh_3d(data[CONNECTED_COMPONENTS_KEY][key])
        return out
    
    @staticmethod
    def write_connected_components(connected_components, data, force_overwrite=False):
        """ Writes a list of connected components """
        if CONNECTED_COMPONENTS_KEY in list(data.keys()):
            if force_overwrite:
                del data[CONNECTED_COMPONENTS_KEY]
            else:
                logging.warning('Connected components already exist, aborting')
                return False
        cc_group = data.create_group(CONNECTED_COMPONENTS_KEY)
        for idx, mesh in enumerate(connected_components):
            one_cc_group = cc_group.create_group(str(idx))
            Hdf5ObjectFactory.write_mesh_3d(mesh, one_cc_group)
        return True
            
    @staticmethod
    def object_metadata(data, metadata_types):
        """ Returns a dictionary of the metadata for the given object """
        if METADATA_KEY not in list(data.keys()):
            return {}
        out = {}
        agg_exist = list(data[METADATA_KEY].keys()) + list(data[METADATA_KEY].attrs.keys())
        for key in metadata_types:
            if key not in agg_exist:
                continue
            metadata_type = metadata_types[key].attrs[METADATA_TYPE_KEY]
            if metadata_type == 'float':
                out[key] = data[METADATA_KEY].attrs[key]
            elif metadata_type == 'arr':
                out[key] = np.asarray(data[METADATA_KEY][key])
        return out
    
    @staticmethod
    def write_object_metadata(metadata_dict, data, metadata_types, force_overwrite=False):
        """ Writes metadata to HDF5 group for object """
        if METADATA_KEY not in list(data.keys()):
            data.create_group(METADATA_KEY)
        metadata_group = data[METADATA_KEY]
        for key, value in metadata_dict.items():
            if metadata_types[key] == 'float':
                if key not in list(metadata_group.attrs.keys()):
                    metadata_group.attrs[key] = value
                elif force_overwrite:
                    metadata_group.attrs.create(key, value)
                else:
                    logging.warning("Metadata {} already exists and overwrite not requested, aborting".format(key))
                    return None
            elif metadata_types[key] == 'arr':
                if key not in metadata_group.keys:
                    metadata_group[key] = value
                elif force_overwrite:
                    metadata_group
                else:
                    logging.warning("Metadata {} already exists and overwrite not requested, aborting".format(key))
                    return None
                
