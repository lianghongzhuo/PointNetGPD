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
Computation of robust grasp quality metrics using random variables.
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

import autolab_core.random_variables as rvs
from dexnet.grasping import PointGraspMetrics3D
from dexnet.learning import MaxIterTerminationCondition, GaussianUniformAllocationMean, RandomContinuousObjective

import IPython


class QuasiStaticGraspQualityRV(rvs.RandomVariable):
    """ RV class for grasp quality on an object.

    Attributes
    ----------
    grasp_rv : :obj:`ParallelJawGraspPoseGaussianRV`
        random variable for gripper pose
    obj_rv : :obj:`GraspableObjectPoseGaussianRV`
        random variable for object pose
    params_rv : :obj:`ParamsGaussianRV`
        random variable for a set of grasp quality parameters
    quality_config : :obj:`GraspQualityConfig`
        parameters for grasp quality computation
    """

    def __init__(self, grasp_rv, obj_rv, params_rv, quality_config):
        self.grasp_rv_ = grasp_rv
        self.obj_rv_ = obj_rv
        self.params_rv_ = params_rv  # samples extra params for quality

        self.sample_count_ = 0
        self.quality_config_ = quality_config

        # preallocation not available
        rvs.RandomVariable.__init__(self, num_prealloc_samples=0)

    @property
    def obj(self):
        return self.graspable_rv_.obj

    @property
    def grasp(self):
        return self.grasp_rv_.grasp

    def sample(self, size=1):
        """ Samples deterministic quasi-static point grasp quality metrics.

        Parameters
        ----------
        size : int
            number of samples to take
        """
        # sample grasp
        cur_time = time.time()
        grasp_sample = self.grasp_rv_.rvs(size=1, iteration=self.sample_count_)
        grasp_time = time.time()

        # sample object
        obj_sample = self.obj_rv_.rvs(size=1, iteration=self.sample_count_)
        obj_time = time.time()

        # sample params
        params_sample = None
        if self.params_rv_ is not None:
            params_sample = self.params_rv_.rvs(size=1, iteration=self.sample_count_)
        params_time = time.time()

        logging.debug('Sampling took %.3f sec' % (params_time - cur_time))

        # compute deterministic quality
        start = time.time()
        q = PointGraspMetrics3D.grasp_quality(grasp_sample, obj_sample,
                                              params_sample)
        quality_time = time.time()

        logging.debug('Quality comp took %.3f sec' % (quality_time - start))

        self.sample_count_ = self.sample_count_ + 1
        return q


class RobustPointGraspMetrics3D:
    """ Class to wrap functions for robust quasistatic point grasp quality metrics.
    """

    @staticmethod
    def expected_quality(grasp_rv, graspable_rv, params_rv, quality_config):
        """
        Compute robustness, or the expected grasp quality wrt given random variables.
        
        Parameters
        ----------
        grasp_rv : :obj:`ParallelJawGraspPoseGaussianRV`
            random variable for gripper pose
        obj_rv : :obj:`GraspableObjectPoseGaussianRV`
            random variable for object pose
        params_rv : :obj:`ParamsGaussianRV`
            random variable for a set of grasp quality parameters
        quality_config : :obj:`GraspQualityConfig`
            parameters for grasp quality computation

        Returns
        -------
        float
            mean quality
        float
            variance of quality samples
        """
        # set up random variable
        q_rv = QuasiStaticGraspQualityRV(grasp_rv, graspable_rv,
                                         params_rv, quality_config)
        candidates = [q_rv]

        # brute force with uniform allocation
        snapshot_rate = quality_config['sampling_snapshot_rate']
        num_samples = quality_config['num_quality_samples']
        objective = RandomContinuousObjective()
        ua = GaussianUniformAllocationMean(objective, candidates)
        ua_result = ua.solve(termination_condition=MaxIterTerminationCondition(num_samples),
                             snapshot_rate=snapshot_rate)

        # convert to estimated prob success
        final_model = ua_result.models[-1]
        mn_q = final_model.means
        std_q = final_model.sample_vars
        return mn_q[0], std_q[0]
