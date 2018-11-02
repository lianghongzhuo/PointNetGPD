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
Configurations for grasp quality computation.
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
import sys
import time

import IPython

# class GraspQualityConfig(object, metaclass=ABCMeta):
class GraspQualityConfig(object):
    """
    Base wrapper class for parameters used in grasp quality computation.
    Used to elegantly enforce existence and type of required parameters.

    Attributes
    ----------
    config : :obj:`dict`
        dictionary mapping parameter names to parameter values
    """
    __metaclass__ = ABCMeta
    def __init__(self, config):
        # check valid config
        self.check_valid(config)

        # parse config
        for key, value in list(config.items()):
            setattr(self, key, value)

    def contains(self, key):
        """ Checks whether or not the key is supported """
        if key in list(self.__dict__.keys()):
            return True
        return False

    def __getattr__(self, key):
        if self.contains(key):
            return object.__getattribute__(self, key)
        return None

    def __getitem__(self, key):
        if self.contains(key):
            return object.__getattribute__(self, key)
        raise KeyError('Key %s not found' %(key))
    
    def keys(self):
        return list(self.__dict__.keys())

    @abstractmethod
    def check_valid(self, config):
        """ Raise an exception if the config is missing required keys """
        pass

class QuasiStaticGraspQualityConfig(GraspQualityConfig):
    """
    Parameters for quasi-static grasp quality computation.

    Attributes
    ----------
    config : :obj:`dict`
        dictionary mapping parameter names to parameter values

    Notes
    -----
    Required configuration key-value pairs in Other Parameters.

    Other Parameters
    ----------------
    quality_method : :obj:`str`
        string name of quasi-static quality metric
    friction_coef : float
        coefficient of friction at contact point
    num_cone_faces : int
        number of faces to use in friction cone approximation
    soft_fingers : bool
        whether to use a soft finger model
    quality_type : :obj:`str`
        string name of grasp quality type (e.g. quasi-static, robust quasi-static)
    check_approach : bool
        whether or not to check the approach direction
    """
    REQUIRED_KEYS = ['quality_method',
                     'friction_coef',
                     'num_cone_faces',
                     'soft_fingers',
                     'quality_type',
                     'check_approach',
                     'all_contacts_required']

    def __init__(self, config):
        GraspQualityConfig.__init__(self, config)

    def __copy__(self):
        """ Makes a copy of the config """
        obj_copy = QuasiStaticGraspQualityConfig(self.__dict__)
        return obj_copy

    def check_valid(self, config):
        for key in QuasiStaticGraspQualityConfig.REQUIRED_KEYS:
            if key not in list(config.keys()):
                raise ValueError('Invalid configuration. Key %s must be specified' %(key))

class RobustQuasiStaticGraspQualityConfig(GraspQualityConfig):
    """
    Parameters for quasi-static grasp quality computation.

    Attributes
    ----------
    config : :obj:`dict`
        dictionary mapping parameter names to parameter values

    Notes
    -----
    Required configuration key-value pairs in Other Parameters.

    Other Parameters
    ----------------
    quality_method : :obj:`str`
        string name of quasi-static quality metric
    friction_coef : float
        coefficient of friction at contact point
    num_cone_faces : int
        number of faces to use in friction cone approximation
    soft_fingers : bool
        whether to use a soft finger model
    quality_type : :obj:`str`
        string name of grasp quality type (e.g. quasi-static, robust quasi-static)
    check_approach : bool
        whether or not to check the approach direction
    num_quality_samples : int
        number of samples to use
    """
    ROBUST_REQUIRED_KEYS = ['num_quality_samples']

    def __init__(self, config):
        GraspQualityConfig.__init__(self, config)

    def __copy__(self):
        """ Makes a copy of the config """
        obj_copy = RobustQuasiStaticGraspQualityConfig(self.__dict__)
        return obj_copy
        
    def check_valid(self, config):
        required_keys = QuasiStaticGraspQualityConfig.REQUIRED_KEYS + \
            RobustQuasiStaticGraspQualityConfig.ROBUST_REQUIRED_KEYS
        for key in required_keys:
            if key not in list(config.keys()):
                raise ValueError('Invalid configuration. Key %s must be specified' %(key))        

class GraspQualityConfigFactory:
    """ Helper class to automatically create grasp quality configurations of different types. """
    @staticmethod
    def create_config(config):
        """ Automatically create a quality config from a dictionary.

        Parameters
        ----------
        config : :obj:`dict`
            dictionary mapping parameter names to parameter values
        """
        if config['quality_type'] == 'quasi_static':
            return QuasiStaticGraspQualityConfig(config)
        elif config['quality_type'] == 'robust_quasi_static':
            return RobustQuasiStaticGraspQualityConfig(config)
        else:
            raise ValueError('Quality config type %s not supported' %(config['type']))
