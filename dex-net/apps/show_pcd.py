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


def show_obj(surface_points_, title, color='b'):
    mlab.figure(bgcolor=(0, 0, 0), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
    if color == 'b':
        color_f = (0, 0, 1)
    elif color == 'r':
        color_f = (1, 0, 0)
    elif color == 'g':
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    points_ = surface_points_
    mlab.points3d(points_[:, 0], points_[:, 1], points_[:, 2], color=color_f, scale_factor=.0007)
    mlab.title(title, size=0.5)


path = os.environ['HOME'] + "/code/grasp-pointnet/pointGPD/data/ycb_rgbd/003_cracker_box/clouds"
pcd_files = glob.glob(os.path.join(path, '*.pcd'))
pcd_files.sort()
for i in range(450, len(pcd_files)):
    pcd = pcd_files[i]
    file_name = pcd.split('/')[-1]
    print(file_name)
    points_c = pcl.load(pcd)
    points = points_c.to_array()
    show_obj(points, file_name)
    mlab.show()
