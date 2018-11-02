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
Tests grasping basic functionality
Author: Jeff Mahler
"""
import copy
import IPython
import logging
import numpy as np
import os
import sys
import time
from unittest import TestCase, TestSuite, TextTestRunner

from autolab_core import RigidTransform, YamlConfig
from perception import CameraIntrinsics

from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, GraspableObject3D, UniformGraspSampler, \
    AntipodalGraspSampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper, PointGraspMetrics3D

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.constants import *

CONFIG = YamlConfig(TEST_CONFIG_NAME)


def random_force_closure_test_case(antipodal=False):
    """
    Generates a random contact point pair and surface normal, constraining the points to be antipodal
    if specified and not antipodal otherwise
    """
    scale = 10
    contacts = scale * (np.random.rand(3, 2) - 0.5)

    mu = 0.0
    while mu == 0.0:
        mu = np.random.rand()
    gamma = 0.0
    while gamma == 0.0:
        gamma = np.random.rand()
    num_facets = 3 + 100 * int(np.random.rand())

    if antipodal:
        tangent_cone_scale = mu
        tangent_cone_add = 0
        n0_mult = 1
        n1_mult = 1
    else:
        n0_mult = 2 * (np.random.randint(0, 2) - 0.5)
        n1_mult = 2 * (np.random.randint(0, 2) - 0.5)
        tangent_cone_scale = 10
        tangent_cone_add = mu

        if (n0_mult < 0 or n1_mult < 0) and np.random.rand() > 0.5:
            tangent_cone_scale = mu
            tangent_cone_add = 0

    v = contacts[:, 1] - contacts[:, 0]
    normals = np.array([-v, v]).T
    normals = normals / np.tile(np.linalg.norm(normals, axis=0), [3, 1])

    U, _, _ = np.linalg.svd(normals[:, 0].reshape(3, 1))
    beta = tangent_cone_scale * np.random.rand() + tangent_cone_add
    theta = 2 * np.pi * np.random.rand()
    normals[:, 0] = n0_mult * normals[:, 0] + beta * np.sin(theta) * U[:, 1] + beta * np.cos(theta) * U[:, 2]

    U, _, _ = np.linalg.svd(normals[:, 1].reshape(3, 1))
    beta = tangent_cone_scale * np.random.rand() + tangent_cone_add
    theta = 2 * np.pi * np.random.rand()
    normals[:, 1] = n1_mult * normals[:, 1] + beta * np.sin(theta) * U[:, 1] + beta * np.cos(theta) * U[:, 2]

    normals = normals / np.tile(np.linalg.norm(normals, axis=0), [3, 1])
    return contacts, normals, num_facets, mu, gamma


class GraspTest(TestCase):
    def test_init_grasp(self):
        # random grasp
        g1 = np.random.rand(3)
        g2 = np.random.rand(3)
        x = (g1 + g2) / 2
        v = g2 - g1
        width = np.linalg.norm(v)
        v = v / width
        configuration = ParallelJawPtGrasp3D.configuration_from_params(x, v, width)

        # test init
        random_grasp = ParallelJawPtGrasp3D(configuration)
        read_configuration = random_grasp.configuration
        self.assertTrue(np.allclose(configuration, read_configuration))
        self.assertTrue(np.allclose(x, random_grasp.center))
        self.assertTrue(np.allclose(v, random_grasp.axis))
        self.assertTrue(np.allclose(width, random_grasp.open_width))

        read_g1, read_g2 = random_grasp.endpoints
        self.assertTrue(np.allclose(g1, read_g1))
        self.assertTrue(np.allclose(g2, read_g2))

        # test bad init
        configuration[4] = 1000
        caught_bad_init = False
        try:
            random_grasp = ParallelJawPtGrasp3D(configuration)
        except:
            caught_bad_init = True
        self.assertTrue(caught_bad_init)

    def test_init_graspable(self):
        of = ObjFile(OBJ_FILENAME)
        sf = SdfFile(SDF_FILENAME)
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

    def test_init_gripper(self):
        gripper = RobotGripper.load(GRIPPER_NAME)

    def test_force_closure(self):
        of = ObjFile(OBJ_FILENAME)
        sf = SdfFile(SDF_FILENAME)
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

        for i in range(NUM_TEST_CASES):
            contacts, normals, _, mu, _ = random_force_closure_test_case(antipodal=True)
            c1 = Contact3D(obj, contacts[:, 0])
            c1.normal = normals[:, 0]
            c2 = Contact3D(obj, contacts[:, 1])
            c2.normal = normals[:, 1]
            self.assertTrue(PointGraspMetrics3D.force_closure(c1, c2, mu, use_abs_value=False))

        for i in range(NUM_TEST_CASES):
            contacts, normals, _, mu, _ = random_force_closure_test_case(antipodal=False)
            c1 = Contact3D(obj, contacts[:, 0])
            c1.normal = normals[:, 0]
            c2 = Contact3D(obj, contacts[:, 1])
            c2.normal = normals[:, 1]
            self.assertFalse(PointGraspMetrics3D.force_closure(c1, c2, mu, use_abs_value=False))

    def test_wrench_in_positive_span(self):
        # simple test for in positive span
        wrench_basis = np.eye(6)
        force_limit = 1000
        num_fingers = 1

        # truly in span, with force limits
        for i in range(NUM_TEST_CASES):
            target_wrench = np.random.rand(6)
            in_span, norm = PointGraspMetrics3D.wrench_in_positive_span(wrench_basis,
                                                                        target_wrench,
                                                                        force_limit,
                                                                        num_fingers)
            self.assertTrue(in_span)

        # not in span, but within force limits
        for i in range(NUM_TEST_CASES):
            target_wrench = -np.random.rand(6)
            in_span, norm = PointGraspMetrics3D.wrench_in_positive_span(wrench_basis,
                                                                        target_wrench,
                                                                        force_limit,
                                                                        num_fingers)
            self.assertFalse(in_span)

        # truly in span, but not with force limits
        force_limit = 0.1
        for i in range(NUM_TEST_CASES):
            target_wrench = np.random.rand(6)
            target_wrench[0] = 1000
            in_span, norm = PointGraspMetrics3D.wrench_in_positive_span(wrench_basis,
                                                                        target_wrench,
                                                                        force_limit,
                                                                        num_fingers)
            self.assertFalse(in_span)

    def test_min_norm_vector_in_facet(self):
        # zero in facet
        facet = np.c_[np.eye(6), -np.eye(6)]
        min_norm, _ = PointGraspMetrics3D.min_norm_vector_in_facet(facet)
        self.assertLess(min_norm, 1e-5)

        # simplex
        facet = np.c_[np.eye(6)]
        min_norm, v = PointGraspMetrics3D.min_norm_vector_in_facet(facet)
        true_v = (1.0 / 6.0) * np.ones(6)
        self.assertTrue(np.allclose(min_norm, np.linalg.norm(true_v)))
        self.assertTrue(np.allclose(v, true_v))

        # single data point, edge case
        facet = np.ones([6, 1])
        min_norm, v = PointGraspMetrics3D.min_norm_vector_in_facet(facet)
        self.assertTrue(np.allclose(min_norm, np.linalg.norm(facet)))
        self.assertTrue(np.allclose(v, facet))

    def test_antipodal_grasp_sampler(self):
        of = ObjFile(OBJ_FILENAME)
        sf = SdfFile(SDF_FILENAME)
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

        gripper = RobotGripper.load(GRIPPER_NAME)

        ags = AntipodalGraspSampler(gripper, CONFIG)
        grasps = ags.generate_grasps(obj, target_num_grasps=NUM_TEST_CASES)

        # test with raw force closure function
        for i, grasp in enumerate(grasps):
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                self.assertTrue(PointGraspMetrics3D.force_closure(c1, c2, CONFIG['sampling_friction_coef']))

    def test_grasp_quality_functions(self):
        num_grasps = NUM_TEST_CASES
        of = ObjFile(OBJ_FILENAME)
        sf = SdfFile(SDF_FILENAME)
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

        gripper = RobotGripper.load(GRIPPER_NAME)

        ags = UniformGraspSampler(gripper, CONFIG)
        grasps = ags.generate_grasps(obj, target_num_grasps=num_grasps)

        # test with grasp quality function
        quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['force_closure'])
        quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, quality_config)

        for i, grasp in enumerate(grasps):
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                fn_fc = quality_fn(grasp).quality
                true_fc = PointGraspMetrics3D.force_closure(c1, c2, quality_config.friction_coef)
                self.assertEqual(fn_fc, true_fc)

    def test_contacts(self):
        num_samples = 128
        mu = 0.5
        num_faces = 8
        num_grasps = NUM_TEST_CASES
        of = ObjFile(OBJ_FILENAME)
        sf = SdfFile(SDF_FILENAME)
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

        gripper = RobotGripper.load(GRIPPER_NAME)

        ags = UniformGraspSampler(gripper, CONFIG)
        grasps = ags.generate_grasps(obj, target_num_grasps=num_grasps)

        for grasp in grasps:
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c

                # test friction cones
                fc1_exists, fc1, n1 = c1.friction_cone(num_cone_faces=num_faces,
                                                       friction_coef=mu)
                if fc1_exists:
                    self.assertEqual(fc1.shape[1], num_faces)
                    alpha = np.tile(fc1.T.dot(c1.normal), [3, 1]).T
                    w = np.tile(c1.normal.reshape(3, 1), [1, 8]).T
                    tan_vecs = fc1.T - alpha * w
                    self.assertTrue(np.allclose(np.linalg.norm(tan_vecs, axis=1), mu))

                fc2_exists, fc2, n2 = c2.friction_cone(num_cone_faces=num_faces,
                                                       friction_coef=mu)
                if fc2_exists:
                    self.assertEqual(fc2.shape[1], num_faces)
                    alpha = np.tile(fc2.T.dot(c2.normal), [3, 1]).T
                    w = np.tile(c2.normal.reshape(3, 1), [1, 8]).T
                    tan_vecs = fc2.T - alpha * w
                    self.assertTrue(np.allclose(np.linalg.norm(tan_vecs, axis=1), mu))

                # test reference frames
                T_contact1_obj = c1.reference_frame(align_axes=True)
                self.assertTrue(np.allclose(T_contact1_obj.z_axis, c1.in_direction))
                self.assertTrue(np.allclose(T_contact1_obj.translation, c1.point))
                for i in range(num_samples):
                    theta = 2 * i * np.pi / num_samples
                    v = np.cos(theta) * T_contact1_obj.x_axis + np.sin(theta) * T_contact1_obj.y_axis
                    self.assertLessEqual(v[0], T_contact1_obj.x_axis[0])

                T_contact2_obj = c2.reference_frame(align_axes=True)
                self.assertTrue(np.allclose(T_contact2_obj.z_axis, c2.in_direction))
                self.assertTrue(np.allclose(T_contact2_obj.translation, c2.point))
                for i in range(num_samples):
                    theta = 2 * i * np.pi / num_samples
                    v = np.cos(theta) * T_contact2_obj.x_axis + np.sin(theta) * T_contact2_obj.y_axis
                    self.assertLessEqual(v[0], T_contact2_obj.x_axis[0])

    def test_find_contacts(self):
        of = ObjFile(OBJ_FILENAME)
        sf = SdfFile(SDF_FILENAME)
        mesh = of.read()
        sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

        surface_points, _ = obj.sdf.surface_points(grid_basis=False)
        indices = np.arange(surface_points.shape[0])
        for i in range(NUM_TEST_CASES):
            np.random.shuffle(indices)
            c1 = surface_points[0, :]
            c2 = surface_points[1, :]
            w = np.linalg.norm(c1 - c2) + 1e-2
            g = ParallelJawPtGrasp3D.grasp_from_endpoints(c1, c2, width=w)
            success, c = g.close_fingers(obj)
            if success:
                c1_est, c2_est = c
                np.assertTrue(np.allclose(c1, c1_est, atol=1e-3, rtol=0.1))
                np.assertTrue(np.allclose(c2, c2_est, atol=1e-3, rtol=0.1))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_suite = TestSuite()
    test_suite.addTest(GraspTest('test_init_grasp'))
    test_suite.addTest(GraspTest('test_init_graspable'))
    test_suite.addTest(GraspTest('test_init_gripper'))
    test_suite.addTest(GraspTest('test_force_closure'))
    test_suite.addTest(GraspTest('test_wrench_in_positive_span'))
    test_suite.addTest(GraspTest('test_min_norm_vector_in_facet'))
    test_suite.addTest(GraspTest('test_antipodal_grasp_sampler'))
    test_suite.addTest(GraspTest('test_grasp_quality_functions'))
    test_suite.addTest(GraspTest('test_contacts'))
    test_suite.addTest(GraspTest('test_find_contacts'))
    TextTestRunner(verbosity=2).run(test_suite)
