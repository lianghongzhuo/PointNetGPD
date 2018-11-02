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
Random Variables for grasping
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import copy
import itertools as it
import logging
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')
import numpy as np
import time

import scipy.linalg
import scipy.stats
import sklearn.cluster

from autolab_core import Point, RandomVariable
from autolab_core.utils import skew, deskew

from dexnet.grasping import ParallelJawPtGrasp3D, GraspableObject3D, GraspQualityConfig

import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file

from autolab_core import SimilarityTransform

import IPython

class GraspableObjectPoseGaussianRV(RandomVariable):
    """ Random variable for sampling graspable objects in different poses, to model uncertainty in object registration.x

    Attributes
    ----------
    s_rv : :obj:`scipy.stats.norm`
        Gaussian random variable for object scale
    t_rv : :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variable for object translation
    r_xi_rv : :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variable of object rotations over the Lie Algebra
    R_sample_sigma : 3x3 :obj:`numpy.ndarray`
        rotation from the sampling reference frame to the random variable reference frame (e.g. for use with uncertainty only in the plane of the table)
    """
    def __init__(self, obj, mean_T_obj_world, config):
        self.obj_ = obj
        self.mean_T_obj_world_ = mean_T_obj_world
        self._parse_config(config)

        # translation in the sampling reference frame
        translation_sigma = self.R_sample_sigma_.T.dot(self.mean_T_obj_world_.translation)

        # setup random variables
        if isinstance(mean_T_obj_world, SimilarityTransform):
            self.s_rv_ = scipy.stats.norm(self.mean_T_obj_world_.scale, self.sigma_scale_**2)
        else:
            self.s_rv_ = scipy.stats.norm(1.0, self.sigma_scale_**2)
        self.t_rv_ = scipy.stats.multivariate_normal(translation_sigma, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)
        self.com_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_com_**2)

        RandomVariable.__init__(self, self.num_prealloc_samples_)

    def _parse_config(self, config):
        self.sigma_rot_ = 1e-6
        self.sigma_trans_ = 1e-6
        self.sigma_scale_ = 1e-6
        self.sigma_com_ = 1e-6
        self.R_sample_sigma_ = np.eye(3)
        self.num_prealloc_samples_ = 0

        if config is not None:
            if 'sigma_obj_rot' in list(config.keys()):
                self.sigma_rot_ = config['sigma_obj_rot']
            elif 'sigma_obj_rot_x' in list(config.keys()) and \
                    'sigma_obj_rot_y' in list(config.keys()) and \
                    'sigma_obj_rot_z' in list(config.keys()):
                self.sigma_rot_ = np.diag([config['sigma_obj_rot_x'],
                                           config['sigma_obj_rot_y'],
                                           config['sigma_obj_rot_z']])
            if 'sigma_obj_trans' in list(config.keys()):
                self.sigma_trans_ = config['sigma_obj_trans']
            elif 'sigma_obj_trans_x' in list(config.keys()) and \
                    'sigma_obj_trans_y' in list(config.keys()) and \
                    'sigma_obj_trans_z' in list(config.keys()):
                self.sigma_trans_ = np.diag([config['sigma_obj_trans_x'],
                                           config['sigma_obj_trans_y'],
                                           config['sigma_obj_trans_z']])
            if 'sigma_obj_scale' in list(config.keys()):
                self.sigma_scale_ = config['sigma_obj_scale']
            if 'sigma_obj_com' in list(config.keys()):
                self.sigma_com_ = config['sigma_obj_com']
            if 'R_sample_sigma' in list(config.keys()):
                self.R_sample_sigma_ = config['R_sample_sigma']
            if 'num_prealloc_samples' in list(config.keys()):
                self.num_prealloc_samples_ = config['num_prealloc_samples']

    @property
    def obj(self):
        return self.obj_

    def sample(self, size=1):
        """ Sample random variables from the model.

        Parameters
        ----------
        size : int
            number of sample to take
        
        Returns
        -------
        :obj:`list` of :obj:`GraspableObject3D`
            sampled graspable objects from the pose random variable
        """
        samples = []
        for i in range(size):
            num_consecutive_failures = 0
            prev_len = len(samples)
            while len(samples) == prev_len:
                try:
                    # sample random pose
                    xi = self.r_xi_rv_.rvs(size=1)
                    S_xi = skew(xi)
                    R = self.R_sample_sigma_.dot(scipy.linalg.expm(S_xi).dot(self.R_sample_sigma_.T.dot(self.mean_T_obj_world_.rotation)))
                    s = max(self.s_rv_.rvs(size=1)[0], 0)
                    t = self.R_sample_sigma_.dot(self.t_rv_.rvs(size=1).T).T
                    z = self.R_sample_sigma_.dot(self.com_rv_.rvs(size=1))
                    sample_tf = SimilarityTransform(rotation=R.T,
                                                    translation=t,
                                                    scale=s)
                    z_tf = sample_tf * Point(z, frame=sample_tf.from_frame)
                    z_tf = z_tf.data
                    
                    # transform object by pose
                    obj_sample = self.obj_.transform(sample_tf)
                    obj_sample.mesh.center_of_mass = z_tf
                    samples.append(obj_sample)

                except Exception as e:
                    num_consecutive_failures += 1
                    if num_consecutive_failures > 3:
                        raise

        # not a list if only 1 sample
        if size == 1 and len(samples) > 0:
            return samples[0]
        return samples

class ParallelJawGraspPoseGaussianRV(RandomVariable):
    """ Random variable for sampling grasps in different poses, to model uncertainty in robot repeatability

    Attributes
    ----------
    t_rv : :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variable for grasp translation
    r_xi_rv : :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variable of grasp rotations over the Lie Algebra
    R_sample_sigma : 3x3 :obj:`numpy.ndarray`
        rotation from the sampling reference frame to the random variable reference frame (e.g. for use with uncertainty only in the plane of the table)
    """
    def __init__(self, grasp, config):
        self.grasp_ = grasp
        self._parse_config(config)

        center_sigma = self.R_sample_sigma_.T.dot(grasp.center)
        self.t_rv_ = scipy.stats.multivariate_normal(center_sigma, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)
        self.open_width_rv_ = scipy.stats.norm(grasp.open_width, self.sigma_open_width_**2)
        self.close_width_rv_ = scipy.stats.norm(grasp.close_width, self.sigma_close_width_**2)
        self.approach_rv_ = scipy.stats.norm(grasp.approach_angle, self.sigma_approach_**2)

        RandomVariable.__init__(self, self.num_prealloc_samples_)

    def _parse_config(self, config):
        self.sigma_rot_ = 1e-6
        self.sigma_trans_ = 1e-6
        self.sigma_open_width_ = 1e-6
        self.sigma_close_width_ = 1e-6
        self.sigma_approach_ = 1e-6
        self.R_sample_sigma_ = np.eye(3)
        self.num_prealloc_samples_ = 0

        if config is not None:
            if 'sigma_grasp_rot' in list(config.keys()):
                self.sigma_rot_ = config['sigma_grasp_rot']
            elif 'sigma_grasp_rot_x' in list(config.keys()) and \
                    'sigma_grasp_rot_y' in list(config.keys()) and \
                    'sigma_grasp_rot_z' in list(config.keys()):
                self.sigma_rot_ = np.diag([config['sigma_grasp_rot_x'],
                                           config['sigma_grasp_rot_y'],
                                           config['sigma_grasp_rot_z']])
            if 'sigma_grasp_trans' in list(config.keys()):
                self.sigma_trans_ = config['sigma_grasp_trans']
            elif 'sigma_grasp_trans_x' in list(config.keys()) and \
                    'sigma_grasp_trans_y' in list(config.keys()) and \
                    'sigma_grasp_trans_z' in list(config.keys()):
                self.sigma_trans_ = np.diag([config['sigma_grasp_trans_x'],
                                           config['sigma_grasp_trans_y'],
                                           config['sigma_grasp_trans_z']])

            if 'sigma_gripper_open_width' in list(config.keys()):
                self.sigma_open_width_ = config['sigma_gripper_open_width']
            if 'sigma_gripper_close_width' in list(config.keys()):
                self.sigma_gripper_close_width_ = config['sigma_gripper_close_width']
            if 'sigma_grasp_approach' in list(config.keys()):
                self.sigma_approach_ = config['sigma_grasp_approach']

            if 'R_sample_sigma' in list(config.keys()):
                self.R_sample_sigma_ = config['R_sample_sigma']
            if 'num_prealloc_samples' in list(config.keys()):
                self.num_prealloc_samples_ = config['num_prealloc_samples']

    @property
    def grasp(self):
        return self.grasp_

    def sample(self, size=1):
        """ Sample random variables from the model.

        Parameters
        ----------
        size : int
            number of sample to take
        
        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            sampled grasps in various poses
        """
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)

            axis_sigma = self.R_sample_sigma_.T.dot(self.grasp_.axis)
            v = self.R_sample_sigma_.dot(scipy.linalg.expm(S_xi).dot(axis_sigma))
            t = self.R_sample_sigma_.dot(self.t_rv_.rvs(size=1).T).T
            open_width = max(self.open_width_rv_.rvs(size=1), 0)
            close_width = max(self.close_width_rv_.rvs(size=1), 0)
            approach = self.approach_rv_.rvs(size=1)

            # transform grasp by pose
            grasp_sample = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(t, v, open_width, approach, self.grasp_.jaw_width, close_width))

            samples.append(grasp_sample)

        if size == 1:
            return samples[0]
        return samples

class ParamsGaussianRV(RandomVariable):
    """ Random variable for sampling a Gaussian set of parameters.

    Attributes
    ----------
    rvs : :obj:`dict` mapping string paramter names to :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variables of different paramters
    """
    def __init__(self, params, u_config):
        if not isinstance(params, GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')
        self.params_ = params
        self._parse_config(u_config)

        self.rvs_ = {}
        for param_name, param_rv in self.sigmas_.items():
            self.rvs_[param_name] = scipy.stats.multivariate_normal(param_rv[0], param_rv[1])
        RandomVariable.__init__(self, self.num_prealloc_samples_)

    def _parse_config(self, sigma_params):
        self.sigmas_ = {}
        self.num_prealloc_samples_ = 0

        if sigma_params is not None:
            for key in list(sigma_params.keys()):
                # only parse the "sigmas"
                ind = key.find('sigma')
                if ind == 0 and len(key) > 7 and key[6:] in list(self.params_.keys()):
                    param_name = key[6:]
                    self.sigmas_[param_name] = (self.params_[param_name], sigma_params[key])
            if 'num_prealloc_samples' in list(sigma_params.keys()):
                self.num_prealloc_samples_ = sigma_params['num_prealloc_samples']

    def mean(self):
        return self.params_
        
    def sample(self, size=1):
        """ Sample random variables from the model.

        Parameters
        ----------
        size : int
            number of sample to take
        
        Returns
        -------
        :obj:`list` of :obj:`dict`
            list of sampled dictionaries of parameters
        """
        samples = []
        for i in range(size):
            # sample random force, torque, etc
            params_sample = copy.copy(self.params_)
            for rv_name, rv in self.rvs_.items():
                if rv_name == 'friction_coef':
                    param_sample =  max(rv.rvs(size=1), 0)
                else:
                    param_sample = rv.rvs(size=1)
                params_sample.__setattr__(rv_name, param_sample)
            samples.append(params_sample)

        if size == 1:
            return samples[0]
        return samples

