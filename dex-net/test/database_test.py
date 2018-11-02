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
Tests database basic functionality
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

from perception import CameraIntrinsics, RenderMode

from meshpy.obj_file import ObjFile
from meshpy.mesh_renderer import ViewsphereDiscretizer, VirtualCamera

from dexnet.constants import READ_WRITE_ACCESS
from dexnet.database import Hdf5Database, MeshProcessor, RescalingType
from dexnet.grasping.grasp import ParallelJawPtGrasp3D
from dexnet.constants import *

CONFIG = YamlConfig(TEST_CONFIG_NAME)

class Hdf5DatabaseTest(TestCase):

    @classmethod
    def setUpClass(cls):
        # remove existing databases
        if os.path.exists(TEST_DB_NAME):
            os.remove(TEST_DB_NAME)
        if os.path.exists(ILLEGAL_DB_NAME):
            os.remove(ILLEGAL_DB_NAME)
        if not os.path.exists(TEST_DB_DIR):
            os.mkdir(TEST_DB_DIR)

    def test_illegal_create(self):
        caught_illegal_db = False
        try:
            database = Hdf5Database(ILLEGAL_DB_NAME, access_level=READ_WRITE_ACCESS)
            database.close()
        except:
            caught_illegal_db = True
        self.assertTrue(caught_illegal_db)

    def test_new_database_and_graspable(self):
        # new database
        database = Hdf5Database(TEST_DB_NAME, access_level=READ_WRITE_ACCESS)
        database.close()
        self.assertTrue(database is not None)

        # new dataset
        database = Hdf5Database(TEST_DB_NAME, access_level=READ_WRITE_ACCESS)
        database.create_dataset(TEST_DS_NAME)
        database.close()
        self.assertTrue(database is not None)

        # read existing dataset
        database = Hdf5Database(TEST_DB_NAME, access_level=READ_WRITE_ACCESS)
        dataset = database.dataset(TEST_DS_NAME)
        self.assertTrue(database is not None and dataset is not None)
        
        # create graspable
        mass = 1.0
        CONFIG['obj_rescaling_type'] = RescalingType.RELATIVE
        mesh_processor = MeshProcessor(OBJ_FILENAME, CONFIG['cache_dir'])
        mesh_processor.generate_graspable(CONFIG)
        dataset.create_graspable(mesh_processor.key, mesh_processor.mesh,
                                 mesh_processor.sdf,
                                 mesh_processor.stable_poses,
                                 mass=mass)

        # read graspable and ensure data integrity
        obj = dataset[mesh_processor.key]
        self.assertTrue(obj.key == mesh_processor.key)
        write_vertices = mesh_processor.mesh.vertices
        write_triangles = mesh_processor.mesh.triangles
        write_sdf_data = mesh_processor.sdf.data
        write_stable_poses = mesh_processor.stable_poses
        load_vertices = obj.mesh.vertices
        load_triangles = obj.mesh.triangles
        load_sdf_data = obj.sdf.data
        load_stable_poses = dataset.stable_poses(obj.key)
        self.assertTrue(np.allclose(write_vertices, load_vertices))
        self.assertTrue(np.allclose(write_triangles, load_triangles))
        self.assertTrue(np.allclose(write_sdf_data, load_sdf_data))        
        self.assertTrue(obj.mass == mass)

        for wsp, lsp in zip(write_stable_poses, load_stable_poses):
            self.assertTrue(np.allclose(wsp.r, lsp.r))
            self.assertTrue(np.allclose(wsp.p, lsp.p))

        self.assertTrue(database is not None and dataset is not None)        

        # test loop access
        for obj in dataset:
            key = obj.key

        # test direct access
        obj = dataset[key]
        self.assertTrue(obj.key == key)

        # read / write meshing
        obj = dataset[dataset.object_keys[0]]        
        mesh_filename = dataset.obj_mesh_filename(obj.key, overwrite=True)
        f = ObjFile(mesh_filename)
        load_mesh = f.read()

        write_vertices = np.array(obj.mesh.vertices)
        write_triangles = np.array(obj.mesh.triangles)
        load_vertices = np.array(load_mesh.vertices)
        load_triangles = np.array(load_mesh.triangles)

        self.assertTrue(np.allclose(write_vertices, load_vertices, atol=1e-5))
        self.assertTrue(np.allclose(write_triangles, load_triangles, atol=1e-5))

        # test rendering images
        stable_poses = dataset.stable_poses(obj.key)
        stable_pose = stable_poses[0]

        # setup virtual camera
        width = CONFIG['width']
        height = CONFIG['height']
        f = CONFIG['focal']
        cx = float(width) / 2
        cy = float(height) / 2
        ci = CameraIntrinsics('camera', fx=f, fy=f, cx=cx, cy=cy,
                              height=height, width=width)
        vp = ViewsphereDiscretizer(min_radius=CONFIG['min_radius'],
                                   max_radius=CONFIG['max_radius'],
                                   num_radii=CONFIG['num_radii'],
                                   min_elev=CONFIG['min_elev']*np.pi,
                                   max_elev=CONFIG['max_elev']*np.pi,
                                   num_elev=CONFIG['num_elev'],
                                   num_az=CONFIG['num_az'],
                                   num_roll=CONFIG['num_roll'])
        vc = VirtualCamera(ci)

        # render segmasks and depth
        render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH, RenderMode.SCALED_DEPTH]
        for render_mode in render_modes:
            rendered_images = vc.wrapped_images_viewsphere(obj.mesh, vp,
                                                           render_mode,
                                                           stable_pose)
            pre_store_num_images = len(rendered_images)
            dataset.store_rendered_images(obj.key, rendered_images,
                                          stable_pose_id=stable_pose.id,
                                          render_mode=render_mode)
            rendered_images = dataset.rendered_images(obj.key,
                                                      stable_pose_id=stable_pose.id,
                                                      render_mode=render_mode)
            post_store_num_images = len(rendered_images)
            self.assertTrue(pre_store_num_images == post_store_num_images)

        # test read / write grasp metrics
        metric_name = CONFIG['metrics'].keys()[0]
        metric_config = CONFIG['metrics'][metric_name]
        
        dataset.create_metric(metric_name, metric_config)
        load_metric_config = dataset.metric(metric_name)
        self.assertTrue(dataset.has_metric(metric_name))
        for key, value in metric_config.iteritems():
            if isinstance(value, dict):
                for k, v in value.iteritems():
                    self.assertTrue(load_metric_config[key][k] == v)
            else:
                self.assertTrue(load_metric_config[key] == value)                

        dataset.delete_metric(metric_name)
        self.assertFalse(dataset.has_metric(metric_name))

        # test read / write grasps
        num_grasps = NUM_DB_GRASPS
        database = Hdf5Database(TEST_DB_NAME, access_level=READ_WRITE_ACCESS)
        dataset = database.dataset(TEST_DS_NAME)
        key = dataset.object_keys[0]

        grasps = []
        grasp_metrics = {}
        for i in range(num_grasps):
            configuration = np.random.rand(9)
            configuration[3:6] = configuration[3:6] / np.linalg.norm(configuration[3:6])
            random_grasp = ParallelJawPtGrasp3D(configuration)
            grasps.append(random_grasp)

        dataset.store_grasps(key, grasps)
        loaded_grasps = dataset.grasps(key)
        for g in loaded_grasps:
            grasp_metrics[g.id] = {}
            grasp_metrics[g.id]['force_closure'] = 1 * (np.random.rand() > 0.5)

        for g1, g2 in zip(grasps, loaded_grasps):
            self.assertTrue(np.allclose(g1.configuration, g2.configuration))

        self.assertTrue(dataset.has_grasps(key))

        dataset.store_grasp_metrics(key, grasp_metrics)
        loaded_grasp_metrics = dataset.grasp_metrics(key, loaded_grasps)
        for i, metrics in loaded_grasp_metrics.iteritems():
            self.assertTrue(metrics['force_closure'] == grasp_metrics[i]['force_closure'])

        # remove grasps            
        dataset.delete_grasps(key)
        self.assertFalse(dataset.has_grasps(key))
        
        # remove rendered images
        render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH, RenderMode.SCALED_DEPTH]
        for render_mode in render_modes:
            dataset.delete_rendered_images(key, stable_pose_id=stable_pose.id,
                                           render_mode=render_mode)

            rendered_images = dataset.rendered_images(key,
                                                      stable_pose_id=stable_pose.id,
                                                      render_mode=render_mode)
            self.assertTrue(len(rendered_images) == 0)

        # remove graspable    
        database = Hdf5Database(TEST_DB_NAME, access_level=READ_WRITE_ACCESS)
        dataset = database.dataset(TEST_DS_NAME)
        key = dataset.object_keys[0]
        dataset.delete_graspable(key)

        obj_deleted = False
        try:
            obj = dataset[key]
        except:
            obj_deleted = True
        self.assertTrue(obj_deleted)

        database.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_suite = TestSuite()
    test_suite.addTest(Hdf5DatabaseTest('test_illegal_create'))
    test_suite.addTest(Hdf5DatabaseTest('test_new_database_and_graspable'))
    TextTestRunner(verbosity=2).run(test_suite)
    
