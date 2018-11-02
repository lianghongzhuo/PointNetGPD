"""
Regressive test for stable poses. Qualitative only.
Author: Jeff Mahler
"""
import IPython
import numpy as np
import os
import random
import sys

from autolab_core import Point, RigidTransform
from meshpy import ObjFile, Mesh3D
from visualization import Visualizer3D as vis

if __name__ == '__main__':
    mesh_name = sys.argv[1]

    #np.random.seed(111)
    #random.seed(111)

    # read mesh
    mesh = ObjFile(mesh_name).read()

    mesh.vertices_ = np.load('../dex-net/data/meshes/lego_vertices.npy')
    mesh.center_of_mass = np.load('../dex-net/data/meshes/lego_com.npy')

    #T_obj_table = RigidTransform(rotation=[0.92275663,  0.13768089,  0.35600924, -0.05311874],
    #                             from_frame='obj', to_frame='table')
    T_obj_table = RigidTransform(rotation=[-0.1335021, 0.87671711, 0.41438141, 0.20452958],
                                 from_frame='obj', to_frame='table')

    stable_pose = mesh.resting_pose(T_obj_table)
    #print stable_pose.r

    table_dim = 0.3
    T_obj_table_plot = mesh.get_T_surface_obj(T_obj_table)
    T_obj_table_plot.translation[0] += 0.1
    vis.figure()
    vis.mesh(mesh, T_obj_table_plot, 
             color=(1,0,0), style='wireframe')
    vis.points(Point(mesh.center_of_mass, 'obj'), T_obj_table_plot,
               color=(1,0,1), scale=0.01)
    vis.pose(T_obj_table_plot, alpha=0.1)
    vis.mesh_stable_pose(mesh, stable_pose, dim=table_dim,
                         color=(0,1,0), style='surface')
    vis.pose(stable_pose.T_obj_table, alpha=0.1)
    vis.show()
    exit(0)

    # compute stable poses
    vis.figure()
    vis.mesh(mesh, color=(1,1,0), style='surface')
    vis.mesh(mesh.convex_hull(), color=(1,0,0))

    stable_poses = mesh.stable_poses()
    
    vis.show()
