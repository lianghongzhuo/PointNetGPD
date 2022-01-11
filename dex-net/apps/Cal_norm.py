#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 09/06/2018 7:47 PM
# File Name  : Cal_norm.py
import os
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D
import matplotlib.pyplot as plt
import numpy as np
import pcl
import multiprocessing
import time
from mayavi import mlab


def show_obj(surface_points_, color="b"):
    if color == "b":
        color_f = (0, 0, 1)
    elif color == "r":
        color_f = (1, 0, 0)
    elif color == "g":
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    points = surface_points_
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=color_f, scale_factor=.0007)


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count("/") == file_dir_.count("/")+1:
            file_list.append(root)
        # print(dirs)  # all the directories in current path
        # print(files)  # all the files in current path
    file_list.sort()
    return file_list


def show_grasp_norm(grasp_bottom_center, grasp_axis):
    un1 = grasp_bottom_center - 0.25 * grasp_axis * 0.25
    un2 = grasp_bottom_center  # - 0.25 * grasp_axis * 0.05
    mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=(0, 1, 0), tube_radius=0.0005)


def show_pcl_norm(grasp_bottom_center, normal_, color="r", clear=False):
    if clear:
        plt.figure()
        plt.clf()
        plt.gcf()
        plt.ion()

    ax = plt.gca(projection="3d")
    un1 = grasp_bottom_center + 0.5 * normal_ * 0.05
    ax.scatter(un1[0], un1[1], un1[2], marker="x", c=color)
    un2 = grasp_bottom_center
    ax.scatter(un2[0], un2[1], un2[2], marker="^", c="g")
    ax.plot([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], "b-", linewidth=1)  # bi normal


def do_job(job_i):
    ii = np.random.choice(all_p.shape[0])
    show_grasp_norm(all_p[ii], surface_normal[ii])
    print("done job", job_i, ii)


if __name__ == "__main__":
    file_dir = os.environ["PointNetGPD_FOLDER"] + "/PointNetGPD/data/ycb-tools/models/ycb"
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7))
    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()
    i = 1  # index of objects to define which object to show
    if os.path.exists(str(file_list_all[i]) + "/google_512k/nontextured.obj"):
        of = ObjFile(str(file_list_all[i]) + "/google_512k/nontextured.obj")
        sf = SdfFile(str(file_list_all[i]) + "/google_512k/nontextured.sdf")

    else:
        print("can not find any obj or sdf file!")
        raise NameError("can not find any obj or sdf file!")
    mesh = of.read()
    sdf = sf.read()
    graspable = GraspableObject3D(sdf, mesh)
    print("Log: opened object")
    begin_time = time.time()
    surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
    all_p = surface_points
    method = "voxel"
    if method == "random":
        surface_points = surface_points[np.random.choice(surface_points.shape[0], 1000, replace=False)]
        surface_normal = []
    elif method == "voxel":
        surface_points = surface_points.astype(np.float32)
        p = pcl.PointCloud(surface_points)
        voxel = p.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * 5] * 3))
        surface_points = voxel.filter().to_array()

        # cal normal with pcl
        use_voxel = False  # use voxelized point to get norm is not accurate
        if use_voxel:
            norm = voxel.filter().make_NormalEstimation()
        else:
            norm = p.make_NormalEstimation()

        norm.set_KSearch(10)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        surface_gg = norm.compute().to_array()[:, 3]
        use_pcl = True
        if use_pcl:
            # for ii in range(surface_normal[0]):
            show_grasp_norm(all_p[0], surface_normal[0])
            # FIXME: multi processing is not working, Why?
            # mayavi do not support
            # cores = multiprocessing.cpu_count()
            # pool = multiprocessing.Pool(processes=cores)
            # pool.map(multiprocessing_jobs.do_job, range(2000))
            sample_points = 500
            for _ in range(sample_points):
                do_job(_)
            mlab.pipeline.surface(mlab.pipeline.open(str(file_list_all[i]) + "/google_512k/nontextured.ply")
                                  , opacity=1)
            mlab.show()
            print(time.time() - begin_time)
            # show_obj(all_p)
    else:
        raise ValueError("No such method", method)

    use_meshpy = False
    if use_meshpy:
        normal = []
        # show norm
        surface_points = surface_points[:100]
        for ind in range(len(surface_points)):
            # use meshpy cal norm:
            p_grid = graspable.sdf.transform_pt_obj_to_grid(surface_points[ind])
            normal_tmp = graspable.sdf.surface_normal(p_grid)
            # use py pcl cal norm, Wrong.
            # normal_tmp = surface_normal[ind]
            if normal_tmp is not None:
                normal.append(normal_tmp)
                show_grasp_norm(surface_points[ind], normal_tmp)
            else:
                print(len(normal))
        mlab.pipeline.surface(mlab.pipeline.open(str(file_list_all[i]) + "/google_512k/nontextured.ply"))
        mlab.show()
