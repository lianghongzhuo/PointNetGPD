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
# Dex-Net 3D visualizer extension
# Author: Jeff Mahler and Jacky Liang
# """
import copy
import json
import IPython
import logging
import numpy as np
import os

try:
    import mayavi.mlab as mv
    import mayavi.mlab as mlab
except:
    logging.info('Failed to import mayavi')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.spatial.distance as ssd

from autolab_core import RigidTransform, Point
from dexnet.grasping import RobotGripper
from visualization import Visualizer3D

from meshpy import StablePose
import meshpy.obj_file as objf
import meshpy.mesh as m


class DexNetVisualizer3D(Visualizer3D):
    """
    Dex-Net extension of the base Mayavi-based 3D visualization tools
    """

    @staticmethod
    def gripper(gripper, grasp, T_obj_world, color=(0.5, 0.5, 0.5)):
        """ Plots a robotic gripper in a pose specified by a particular grasp object.

        Parameters
        ----------
        gripper : :obj:`dexnet.grasping.RobotGripper`
            the gripper to plot
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot the gripper performing
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        color : 3-tuple
            color of the gripper mesh
        """
        T_gripper_obj = grasp.gripper_pose(gripper)
        T_gripper_world = T_obj_world * T_gripper_obj
        T_mesh_world = T_gripper_world * gripper.T_mesh_gripper.inverse()
        T_mesh_world = T_mesh_world.as_frames('obj', 'world')
        Visualizer3D.mesh(gripper.mesh.trimesh, T_mesh_world, style='surface', color=color)

    @staticmethod
    def grasp(grasp, T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),
              tube_radius=0.002, endpoint_color=(0, 1, 0),
              endpoint_scale=0.004, grasp_axis_color=(0, 1, 0)):
        """ Plots a grasp as an axis and center.

        Parameters
        ----------
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        tube_radius : float
            radius of the plotted grasp axis
        endpoint_color : 3-tuple
            color of the endpoints of the grasp axis
        endpoint_scale : 3-tuple
            scale of the plotted endpoints
        grasp_axis_color : 3-tuple
            color of the grasp axis
        """
        g1, g2 = grasp.endpoints
        center = grasp.center
        g1 = Point(g1, 'obj')
        g2 = Point(g2, 'obj')
        center = Point(center, 'obj')

        g1_tf = T_obj_world.apply(g1)
        g2_tf = T_obj_world.apply(g2)
        center_tf = T_obj_world.apply(center)
        grasp_axis_tf = np.array([g1_tf.data, g2_tf.data])

        mlab.points3d(g1_tf.data[0], g1_tf.data[1], g1_tf.data[2], color=endpoint_color, scale_factor=endpoint_scale)
        mlab.points3d(g2_tf.data[0], g2_tf.data[1], g2_tf.data[2], color=endpoint_color, scale_factor=endpoint_scale)

        mlab.plot3d(grasp_axis_tf[:, 0], grasp_axis_tf[:, 1], grasp_axis_tf[:, 2], color=grasp_axis_color,
                    tube_radius=tube_radius)

    @staticmethod
    def gripper_on_object(gripper, grasp, obj, stable_pose=None,
                          T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                          gripper_color=(0.5, 0.5, 0.5), object_color=(0.5, 0.5, 0.5),
                          style='surface', plot_table=True, table_dim=0.15):
        """ Visualize a gripper on an object.
        
        Parameters
        ----------
        gripper : :obj:`dexnet.grasping.RobotGripper`
            gripper to plot
        grasp : :obj:`dexnet.grasping.Grasp`
            grasp to plot the gripper in
        obj : :obj:`dexnet.grasping.GraspableObject3D`
            3D object to plot the gripper on
        stable_pose : :obj:`autolab_core.RigidTransform`
            stable pose of the object on a planar worksurface
        T_table_world : :obj:`autolab_core.RigidTransform`
            pose of table, specified as a transformation from mesh frame to world frame
        gripper_color : 3-tuple
            color of the gripper mesh
        object_color : 3-tuple
            color of the object mesh
        style : :obj:`str`
            color of the object mesh
        plot_table : bool
            whether or not to plot the table
        table_dim : float
            dimension of the table
        """
        if stable_pose is None:
            Visualizer3D.mesh(obj.mesh.trimesh, color=object_color, style=style)
            T_obj_world = RigidTransform(from_frame='obj',
                                         to_frame='world')
        else:
            T_obj_world = Visualizer3D.mesh_stable_pose(obj.mesh.trimesh, stable_pose,
                                                        T_table_world=T_table_world,
                                                        color=object_color, style=style,
                                                        plot_table=plot_table, dim=table_dim)
        DexNetVisualizer3D.gripper(gripper, grasp, T_obj_world, color=gripper_color)

