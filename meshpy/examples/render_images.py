"""
Renders an image for a mesh in each stable pose to demo the rendering interface.
Author: Jeff Mahler
"""
import argparse
import copy
import IPython
import logging
import numpy as np
import os
import sys
import time

import autolab_core.utils as utils
from autolab_core import NormalCloud, PointCloud, RigidTransform
from perception import CameraIntrinsics, ObjectRender, RenderMode
from meshpy import MaterialProperties, LightingProperties, ObjFile, VirtualCamera, ViewsphereDiscretizer, SceneObject

from visualization import Visualizer2D as vis
from visualization import Visualizer3D as vis3d

if __name__ == '__main__':
    # parse args
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_filename', type=str, help='filename for .OBJ mesh file to render')
    args = parser.parse_args()

    vis_normals = False

    # read data
    mesh_filename = args.mesh_filename
    _, mesh_ext = os.path.splitext(mesh_filename)
    if mesh_ext != '.obj':
        raise ValueError('Must provide mesh in Wavefront .OBJ format!') 
    orig_mesh = ObjFile(mesh_filename).read()
    mesh = orig_mesh.subdivide(min_tri_length=0.01)
    mesh.compute_vertex_normals()
    stable_poses = mesh.stable_poses()

    if vis_normals:
        vis3d.figure()
        vis3d.mesh(mesh)
        vis3d.normals(NormalCloud(mesh.normals.T), PointCloud(mesh.vertices.T), subsample=10)
        vis3d.show()

    d = utils.sqrt_ceil(len(stable_poses))
    vis.figure(size=(16,16))

    table_mesh = ObjFile('data/meshes/table.obj').read()
    table_mesh = table_mesh.subdivide()
    table_mesh.compute_vertex_normals()
    table_mat_props = MaterialProperties(color=(0,255,0),
                                         ambient=0.5,
                                         diffuse=1.0,
                                         specular=1,
                                         shininess=0)

    for k, stable_pose in enumerate(stable_poses):
        logging.info('Rendering stable pose %d' %(k))

        # set resting pose
        T_obj_world = mesh.get_T_surface_obj(stable_pose.T_obj_table).as_frames('obj', 'world')
    
        # load camera intrinsics
        camera_intr = CameraIntrinsics.load('data/camera_intr/primesense_carmine_108.intr')
        #camera_intr = camera_intr.resize(4)
    
        # create virtual camera
        virtual_camera = VirtualCamera(camera_intr)

        
        # create lighting props
        T_light_camera = RigidTransform(translation=[0,0,0],
                                        from_frame='light',
                                        to_frame=camera_intr.frame)
        light_props = LightingProperties(ambient=-0.25,
                                         diffuse=1,
                                         specular=0.25,
                                         T_light_camera=T_light_camera,
                                         cutoff=180)
        
        # create material props
        mat_props = MaterialProperties(color=(249,241,21),
                                       ambient=0.5,
                                       diffuse=1.0,
                                       specular=1,
                                       shininess=0)

        # create scene objects
        scene_objs = {'table': SceneObject(table_mesh, T_obj_world.inverse(),
                                           mat_props=table_mat_props)}
        for name, scene_obj in scene_objs.items():
            virtual_camera.add_to_scene(name, scene_obj)
        
        # camera pose
        cam_dist = 0.3
        T_camera_world = RigidTransform(rotation=np.array([[0, 1, 0],
                                                           [1, 0, 0],
                                                           [0, 0, -1]]),
                                        translation=[0,0,cam_dist],
                                        from_frame=camera_intr.frame,
                                        to_frame='world')
        
        T_obj_camera = T_camera_world.inverse() * T_obj_world

        # show mesh
        if False:
            vis3d.figure()
            vis3d.mesh(mesh, T_obj_camera)
            vis3d.pose(RigidTransform(), alpha=0.1)
            vis3d.pose(T_obj_camera, alpha=0.1)
            vis3d.show()

        # render depth image
        render_start = time.time()
        IPython.embed()
        renders = virtual_camera.wrapped_images(mesh,
                                                [T_obj_camera],
                                                RenderMode.RGBD_SCENE,
                                                mat_props=mat_props,
                                                light_props=light_props,
                                                debug=False)
        render_stop = time.time()
        logging.info('Render took %.3f sec' %(render_stop-render_start))

        vis.subplot(d,d,k+1)
        vis.imshow(renders[0].image.color)
        #vis.imshow(renders[0].image.depth)
    vis.show()
