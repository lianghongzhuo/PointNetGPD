#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 21/08/2018 9:08 PM 
# File Name  : show_pcd.py
import numpy as np
from mayavi import mlab
import glob
import os
import pcl
import pickle
import logging
logging.getLogger().setLevel(logging.FATAL)
data_path = os.environ["PointNetGPD_FOLDER"] + "/PointNetGPD/data"

def show_obj(points_, ply_name_,obj_transform, color="b"):
    mlab.figure(bgcolor=(0, 0, 0), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
    if color == "b":
        color_f = (0, 0, 1)
    elif color == "r":
        color_f = (1, 0, 0)
    elif color == "g":
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    points_ = np.dot(points_, obj_transform[:3,:3])
    points_ = points_ + obj_transform[:3, 3].reshape(1, 3)
    mlab.points3d(points_[:, 0], points_[:, 1], points_[:, 2], color=color_f, scale_factor=.0007)
    mlab.pipeline.surface(mlab.pipeline.open(ply_name_), opacity=0.8)


def main():
    obj_name = "003_cracker_box"
    obj_name = "004_sugar_box"

    path = data_path + "/ycb_rgbd/{}/clouds".format(obj_name)
    ply_name_ = data_path + "/ycb_meshes_google/{}/google_512k/nontextured.ply".format(obj_name)

    transform = pickle.load(open(data_path + "/google2cloud.pkl", "rb"))
    transform_pkl_obj = transform[obj_name][1]
    print(transform_pkl_obj)
    pcd_files = glob.glob(os.path.join(path, "*.pcd"))
    points = np.array([[0, 0, 0]])
    for i in range(10):
        pcd = pcd_files[i]
        file_name = pcd.split("/")[-1]
        print(file_name)
        points = np.vstack([points, pcl.load(pcd).to_array()])
    show_obj(points[1:], ply_name_, transform_pkl_obj)
    mlab.show()

if __name__ == "__main__":
    main()

