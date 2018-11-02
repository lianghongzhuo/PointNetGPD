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
User-friendly functions for computing grasp quality metrics.
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
import os
import scipy.stats
import sys
import time

from dexnet.grasping import Grasp, GraspableObject, GraspQualityConfig
from dexnet.grasping.robust_grasp_quality import RobustPointGraspMetrics3D
from dexnet.grasping.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, ParamsGaussianRV
from dexnet.grasping import PointGraspMetrics3D

from autolab_core import RigidTransform
import IPython

class GraspQualityResult:
    """ Stores the results of grasp quality computation.

    Attributes
    ----------
    quality : float
        value of quality
    uncertainty : float
        uncertainty estimate of the quality value
    quality_config : :obj:`GraspQualityConfig`
    """
    def __init__(self, quality, uncertainty=0.0, quality_config=None):
        self.quality = quality
        self.uncertainty = uncertainty
        self.quality_config = quality_config            

# class GraspQualityFunction(object, metaclass=ABCMeta):
class GraspQualityFunction(object):
    """
    Abstraction for grasp quality functions to make scripts for labeling with quality functions simple and readable.

    Attributes
    ----------
    graspable : :obj:`GraspableObject3D`
        object to evaluate grasp quality on
    quality_config : :obj:`GraspQualityConfig`
        set of parameters to evaluate grasp quality
    """
    __metaclass__ = ABCMeta


    def __init__(self, graspable, quality_config):
        # check valid types
        if not isinstance(graspable, GraspableObject):
            raise ValueError('Must provide GraspableObject')
        if not isinstance(quality_config, GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')

        # set member variables
        self.graspable_ = graspable
        self.quality_config_ = quality_config

        self._setup()

    def __call__(self, grasp):
        return self.quality(grasp)

    @abstractmethod
    def _setup(self):
        """ Sets up common variables for grasp quality evaluations """
        pass

    @abstractmethod
    def quality(self, grasp):
        """ Compute grasp quality.

        Parameters
        ----------
        grasp : :obj:`GraspableObject3D`
            grasp to quality quality on

        Returns
        -------
        :obj:`GraspQualityResult`
            result of quality computation
        """
        pass
        
class QuasiStaticQualityFunction(GraspQualityFunction):
    """ Grasp quality metric using a quasi-static model.
    """
    def __init__(self, graspable, quality_config):
        GraspQualityFunction.__init__(self, graspable, quality_config)

    @property
    def graspable(self):
        return self.graspable_

    @graspable.setter
    def graspable(self, obj):
        self.graspable_ = obj

    def _setup(self):
        if self.quality_config_.quality_type != 'quasi_static':
            raise ValueError('Quality configuration must be quasi static')

    def quality(self, grasp):
        """ Compute grasp quality using a quasistatic method.

        Parameters
        ----------
        grasp : :obj:`GraspableObject3D`
            grasp to quality quality on

        Returns
        -------
        :obj:`GraspQualityResult`
            result of quality computation
        """
        if not isinstance(grasp, Grasp):
            raise ValueError('Must provide Grasp object to compute quality')

        quality = PointGraspMetrics3D.grasp_quality(grasp, self.graspable_,
                                                    self.quality_config_)
        return GraspQualityResult(quality, quality_config=self.quality_config_)

class RobustQuasiStaticQualityFunction(GraspQualityFunction):
    """ Grasp quality metric using a robust quasi-static model (average over random perturbations)
    """
    def __init__(self, graspable, quality_config, T_obj_world=RigidTransform(from_frame='obj', to_frame='world')):
        self.T_obj_world_ = T_obj_world
        GraspQualityFunction.__init__(self, graspable, quality_config)

    @property
    def graspable(self):
        return self.graspable_

    @graspable.setter
    def graspable(self, obj):
        self.graspable_ = obj
        self._setup()

    def _setup(self):
        if self.quality_config_.quality_type != 'robust_quasi_static':
            raise ValueError('Quality configuration must be robust quasi static')
        self.graspable_rv_ = GraspableObjectPoseGaussianRV(self.graspable_,
                                                           self.T_obj_world_,
                                                           self.quality_config_.obj_uncertainty)
        self.params_rv_ = ParamsGaussianRV(self.quality_config_,
                                           self.quality_config_.params_uncertainty)

    def quality(self, grasp):
        """ Compute grasp quality using a robust quasistatic method.

        Parameters
        ----------
        grasp : :obj:`GraspableObject3D`
            grasp to quality quality on

        Returns
        -------
        :obj:`GraspQualityResult`
            result of quality computation
        """
        if not isinstance(grasp, Grasp):
            raise ValueError('Must provide Grasp object to compute quality')
        grasp_rv = ParallelJawGraspPoseGaussianRV(grasp,
                                                  self.quality_config_.grasp_uncertainty)
        mean_q, std_q = RobustPointGraspMetrics3D.expected_quality(grasp_rv,
                                                                   self.graspable_rv_,
                                                                   self.params_rv_,
                                                                   self.quality_config_)
        return GraspQualityResult(mean_q, std_q, quality_config=self.quality_config_)

class GraspQualityFunctionFactory:
    @staticmethod
    def create_quality_function(graspable, quality_config):
        """ Creates a quality function for a particular object based on a configuration, which can be passed directly from a configuration file.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            object to create quality function for
        quality_config : :obj:`GraspQualityConfig`
            parameters for quality function
        """
        # check valid types
        if not isinstance(graspable, GraspableObject):
            raise ValueError('Must provide GraspableObject')
        if not isinstance(quality_config, GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')
        
        if quality_config.quality_type == 'quasi_static':
            return QuasiStaticQualityFunction(graspable, quality_config)
        elif quality_config.quality_type == 'robust_quasi_static':
            return RobustQuasiStaticQualityFunction(graspable, quality_config)
        else:
            raise ValueError('Grasp quality type %s not supported' %(quality_config.quality_type))
