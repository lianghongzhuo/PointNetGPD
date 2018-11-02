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
from dexnet.grasping.contacts import Contact3D, SurfaceWindow
from dexnet.grasping.graspable_object import GraspableObject, GraspableObject3D
from dexnet.grasping.grasp import Grasp, PointGrasp, ParallelJawPtGrasp3D
from dexnet.grasping.gripper import RobotGripper
from dexnet.grasping.grasp_quality_config import GraspQualityConfig, QuasiStaticGraspQualityConfig, \
    RobustQuasiStaticGraspQualityConfig, GraspQualityConfigFactory
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, \
    ParamsGaussianRV
from dexnet.grasping.robust_grasp_quality import QuasiStaticGraspQualityRV, RobustPointGraspMetrics3D
from dexnet.grasping.grasp_quality_function import GraspQualityResult, GraspQualityFunction, \
    QuasiStaticQualityFunction, RobustQuasiStaticQualityFunction, GraspQualityFunctionFactory

try:
    from dexnet.grasping.collision_checker import OpenRaveCollisionChecker, GraspCollisionChecker
except Exception:
    print('Unable to import OpenRaveCollisionChecker and GraspCollisionChecker! Likely due to missing '
          'OpenRave dependency.')
    print('Install OpenRave 0.9 from source if required. Instructions can be found at '
          'http://openrave.org/docs/latest_stable/coreapihtml/installation_linux.html')

from dexnet.grasping.grasp_sampler import GraspSampler, UniformGraspSampler, GaussianGraspSampler, \
    AntipodalGraspSampler, GpgGraspSampler, PointGraspSampler, GpgGraspSamplerPcl

__all__ = ['Contact3D', 'GraspableObject', 'GraspableObject3D', 'ParallelJawPtGrasp3D',
           'Grasp', 'PointGrasp', 'RobotGripper', 'PointGraspMetrics3D',
           'GraspQualityConfig', 'QuasiStaticGraspQualityConfig', 'RobustQuasiStaticGraspQualityConfig',
           'GraspQualityConfigFactory',
           'GraspSampler', 'UniformGraspSampler', 'GaussianGraspSampler', 'AntipodalGraspSampler',
           'GpgGraspSampler', 'PointGraspSampler', 'GpgGraspSamplerPcl',
           'GraspableObjectPoseGaussianRV', 'ParallelJawGraspPoseGaussianRV', 'ParamsGaussianRV',
           'QuasiStaticGraspQualityRV', 'RobustPointGraspMetrics3D',
           'GraspQualityResult', 'GraspQualityFunction', 'QuasiStaticQualityFunction',
           'RobustQuasiStaticQualityFunction', 'GraspQualityFunctionFactory',
           'OpenRaveCollisionChecker', 'GraspCollisionChecker', ]

# module name spoofing for correct imports
from dexnet.grasping import grasp
import sys
sys.modules['dexnet.grasp'] = grasp
