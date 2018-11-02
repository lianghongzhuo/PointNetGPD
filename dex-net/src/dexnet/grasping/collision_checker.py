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
Collision checking using OpenRAVE
Author: Jeff Mahler
"""
import logging
import time
import numpy as np

USE_OPENRAVE = True
try:
    import openravepy as rave
except:
    logging.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')

import IPython

from autolab_core import RigidTransform

class OpenRaveCollisionChecker(object):
    """ Wrapper for collision checking with OpenRAVE
    """
    env_ = None

    def __init__(self, env=None, view=False, win_height=1200, win_width=1200, cam_dist=0.5):
        """
        Initialize an OpenRaveCollisionChecker

        Parameters
        ----------
        env : :obj:`openravepy.Environment`
            environment to use
        view : bool
            whether or not to open a viewer (does not work when another import has grabbed the Qt thread, e.g. Mayavi or matplotlib)
        win_height : int
            height of view window
        win_width : int
            width of view window
        cam_dist : float
            distance of camera to view window
        """
        if not USE_OPENRAVE:
            raise ValueError('Cannot instantiate OpenRave collision checker')

        if env is None and OpenRaveCollisionChecker.env_ is None:
            OpenRaveCollisionChecker._setup_rave_env()
            
        self._view = view
        if self._view:
            self._init_viewer(win_height, win_width, cam_dist)
            
        self._objs = {}
        self._objs_tf = {}
        
    def remove_object(self, name):
        """ Remove an object from the collision checking environment.

        Parameters
        ----------
        name : :obj:`str`
            name of object to remove
        """
        if name not in self._objs:
            return
        self.env.Remove(self._objs[name])
        self._objs.pop(name)
        self._objs_tf.pop(name)
        
    def set_object(self, name, filename, T_world_obj=None):
        """ Add an object to the collision checking environment.

        Parameters
        ----------
        name : :obj:`str`
            name of object to remove
        filename : :obj:`str`
            filename of triangular mesh (e.g. .STL or .OBJ)
        T_world_obj : :obj:`autolab_core.RigidTransform`
            transformation from object to world frame
        """
        if name in self._objs:
            self.env.Remove(self._objs[name])
        self.env.Load(filename)
        obj = self.env.GetBodies()[-1]
        self._objs[name] = obj
        
        if T_world_obj is None:
            T_world_obj = RigidTransform(from_frame=name, to_frame='world')
        self.set_transform(name, T_world_obj)
    
    def set_transform(self, name, T_world_obj):
        """ Set the pose of an object in the environment.
        
        Parameters
        ----------
        name : :obj:`str`
            name of object to move
        T_world_obj : :obj:`autolab_core.RigidTransform`
            transformation from object to world frame
        """
        T_world_obj_mat = OpenRaveCollisionChecker._tf_to_rave_mat(T_world_obj)
        self._objs[name].SetTransform(T_world_obj_mat)
        self._objs_tf[name] = T_world_obj.copy()
        
    def in_collision_single(self, target_name, names=None):
        """ Checks whether a target object collides with a given set of objects in the environment.
        
        Parameters
        ----------
        target_name : :obj:`str`
            name of target object to check collisions for
        names : :obj:`list` of :obj:`str`
            names of target objects to check collisions with

        Returns
        -------
        bool
            True if a collision occurs, False otherwise
        """
        if self._view and self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
    
        if names is None:
            names = list(self._objs.keys())
            
        target_obj = self._objs[target_name]
        for other_name in names:
            if other_name != target_name:
                if self.env.CheckCollision(self._objs[other_name], target_obj):
                    logging.debug('Collision between: {0} and {1}'.format(other_name, target_name))
                    return True
        
        return False

    def in_collision(self, names=None):
        """ Checks whether there are any pairwise collisions between objects in the environment.
        
        Parameters
        ----------
        names : :obj:`list` of :obj:`str`
            names of target objects to check collisions with

        Returns
        -------
        bool
            True if a collision occurs, False otherwise
        """
        if self._view and self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
    
        if names is None:
            names = list(self._objs.keys())
            
        for name1 in names:
            for name2 in names:
                if name1 != name2:
                    if self.env.CheckCollision(self._objs[name1], self._objs[name2]):
                        logging.debug('Collision between: {0} and {1}'.format(name1, name2))
                        return True
        
        return False
        
    @staticmethod
    def _tf_to_rave_mat(tf):
        """ Convert a RigidTransform to an OpenRAVE matrix """
        position = tf.position
        orientation = tf.quaternion
        pose = np.array([orientation[0], orientation[1], orientation[2], orientation[3], 
                         position[0], position[1], position[2]])
        mat = rave.matrixFromPose(pose)
        return mat
        
    def __del__(self):
        for obj in list(self._objs.values()):
            self.env.Remove(obj)
            
    @property
    def env(self):
        if OpenRaveCollisionChecker.env_ is None:
            OpenRaveCollisionChecker._setup_rave_env()
        return OpenRaveCollisionChecker.env_
        
    def set_view(self, view):
        self._view = view

    @staticmethod
    def _setup_rave_env():
        """ OpenRave environment """
        OpenRaveCollisionChecker.env_ = rave.Environment()
        
    def _init_viewer(self, height, width, cam_dist):
        """ Initialize the OpenRave viewer """
        # set OR viewer
        OpenRaveCollisionChecker.env_.SetViewer("qtcoin")
        viewer = self.env.GetViewer()
        viewer.SetSize(width, height)

        T_cam_obj = np.eye(4)
        R_cam_obj = np.array([[0,  0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
        T_cam_obj[:3,:3] = R_cam_obj
        T_cam_obj[0,3] = -cam_dist
        self.T_cam_obj_ = T_cam_obj

        # set view based on object
        self.T_obj_world_ = np.eye(4)
        self.T_cam_world_ = self.T_obj_world_.dot(self.T_cam_obj_)
        viewer.SetCamera(self.T_cam_world_, cam_dist)

class GraspCollisionChecker(OpenRaveCollisionChecker):
    """ Collision checker that automatcially handles grasp objects.
    """
    def __init__(self, gripper, env=None, view=False, win_height=1200, win_width=1200, cam_dist=0.5):
        """
        Initialize a GraspCollisionChecker.

        Parameters
        ----------
        gripper : :obj:`RobotGripper`
            robot gripper to use for collision checking
        env : :obj:`openravepy.Environment`
            environment to use
        view : bool
            whether or not to open a viewer (does not work when another import has grabbed the Qt thread, e.g. Mayavi or matplotlib)
        win_height : int
            height of view window
        win_width : int
            width of view window
        cam_dist : float
            distance of camera to view window
        """
        OpenRaveCollisionChecker.__init__(self, env, view, win_height, win_width, cam_dist)
        self._gripper = gripper
        self.set_object('gripper', self._gripper.mesh_filename)

    @property
    def obj_names(self):
        """ List of object names """
        return list(self._objs_tf.keys())

    def set_target_object(self, key):
        """ Sets the target graspable object. """
        if key in self.obj_names:
            self._graspable_key = key

    def set_graspable_object(self, graspable, T_obj_world=RigidTransform(from_frame='obj',
                                                                         to_frame='world')):
        """ Adds and sets the target object in the environment.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        """
        self.set_object(graspable.key, graspable.model_name, T_obj_world)
        self.set_target_object(graspable.key)
        
    def add_graspable_object(self, graspable, T_obj_world=RigidTransform(from_frame='obj',
                                                                         to_frame='world')):
        """ Adds the target object to the environment.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to add
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the transformation from obj to world frame
        """
        self.set_object(graspable.key, graspable.model_name, T_obj_world)

    def set_table(self, filename, T_table_world):
        """ Set the table geometry and position in the environment.

        Parameters
        ----------
        filename : :obj:`str`
            name of table mesh file (e.g. .STL or .OBJ)
        T_table_world : :obj:`autolab_core.RigidTransform`
            pose of table w.r.t. world
        """
        self.set_object('table', filename, T_table_world)
                
    def grasp_in_collision(self, T_obj_gripper, key=None):
        """ Check collision of grasp with target object.
        
        Parameters
        ----------
        T_obj_gripper : :obj:`autolab_core.RigidTransform`
            pose of the gripper w.r.t the object
        key : str
            key of object to grasp

        Returns
        -------
        bool
            True if the grasp is in collision, False otherwise
        """
        # set key
        if key is None or key not in list(self._objs_tf.keys()):
            key = self._graspable_key

        # set gripper transformation
        T_world_gripper = self._objs_tf[key] * T_obj_gripper
        T_world_mesh = T_world_gripper * self._gripper.T_mesh_gripper.inverse()
        self.set_transform('gripper', T_world_mesh)

        # check collisions
        return self.in_collision_single('gripper')

    def collides_along_approach(self, grasp, approach_dist, delta_approach,
                                key=None):
        """ Checks whether a grasp collides along its approach direction.
        Currently assumes that the collision checker has loaded the object.
        
        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to check collisions for
        approach_dist : float
            how far back to check along the approach direction
        delta_approach : float
            how finely to discretize poses along the approach direction
        key : str
            key of object to grasp

        Returns
        -------
        bool
            whether or not the grasp is in collision
        """
        # get the gripper pose and axis of approach
        T_grasp_obj = grasp.T_grasp_obj
        grasp_approach_axis = T_grasp_obj.x_axis

        # setup variables
        collides = False
        cur_approach = 0.0
        while cur_approach <= approach_dist and not collides:
            # back up along approach dir
            T_approach_obj = T_grasp_obj.copy()
            T_approach_obj.translation -= cur_approach * grasp_approach_axis
            T_gripper_obj = T_approach_obj * self._gripper.T_grasp_gripper
            
            # check collisions
            collides = self.grasp_in_collision(T_gripper_obj, key=key)
            cur_approach += delta_approach
            
        return collides

