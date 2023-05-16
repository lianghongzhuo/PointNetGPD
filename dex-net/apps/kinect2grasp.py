#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 05/08/2018 6:04 PM
# File Name  : kinect2grasp.py
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import numpy as np
import pointclouds
import voxelgrid
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSamplerPcl
import os
from pyquaternion import Quaternion
import sys
from os import path
from scipy.stats import mode
import multiprocessing as mp
try:
    from gpd_grasp_msgs.msg import GraspConfig
    from gpd_grasp_msgs.msg import GraspConfigList
except ImportError:
    print("Please install grasp msgs from https://github.com/TAMS-Group/gpd_grasp_msgs in your ROS workspace")
    exit()

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath("__file__")))))
sys.path.append(os.environ['PointNetGPD_FOLDER'] + "/PointNetGPD")
from main_test import test_network, model, args
import logging
logging.getLogger().setLevel(logging.FATAL)
# global config:
yaml_config = YamlConfig(os.environ['PointNetGPD_FOLDER'] + "/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
gripper = RobotGripper.load(gripper_name, os.environ['PointNetGPD_FOLDER'] + "/dex-net/data/grippers")
ags = GpgGraspSamplerPcl(gripper, yaml_config)
value_fc = 0.4  # no use, set a random number
num_grasps = 40
num_workers = 20
max_num_samples = 150
n_voxel = 500

minimal_points_send_to_point_net = 20
marker_life_time = 8

show_bad_grasp = False
save_grasp_related_file = False

show_final_grasp = args.show_final_grasp
tray_grasp = args.tray_grasp
using_mp = args.using_mp
single_obj_testing = False  # if True, it will wait for input before get pointcloud
# number of points put into neural network
if args.model_type == "100":  # minimal points send for training
    input_points_num = 500
elif args.model_type == "50":
    input_points_num = 750
elif args.model_type == "3class":
    input_points_num = 500
else:
    input_points_num = 0


def remove_table_points(points_voxel_, vis=False):
    xy_unique = np.unique(points_voxel_[:, 0:2], axis=0)
    new_points_voxel_ = points_voxel_
    pre_del = np.zeros([1])
    for i in range(len(xy_unique)):
        tmp = []
        for j in range(len(points_voxel_)):
            if np.array_equal(points_voxel_[j, 0:2], xy_unique[i]):
                tmp.append(j)
        print(len(tmp))
        if len(tmp) < 3:
            tmp = np.array(tmp)
            pre_del = np.hstack([pre_del, tmp])
    if len(pre_del) != 1:
        pre_del = pre_del[1:]
        new_points_voxel_ = np.delete(points_voxel_, pre_del, 0)
    print("Success delete [[ {} ]] points from the table!".format(len(points_voxel_) - len(new_points_voxel_)))
    return new_points_voxel_


def remove_white_pixel(msg, points_, vis=False):
    points_with_c_ = pointclouds.pointcloud2_to_array(msg)
    points_with_c_ = pointclouds.split_rgb_field(points_with_c_)
    r = np.asarray(points_with_c_['r'], dtype=np.uint32)
    g = np.asarray(points_with_c_['g'], dtype=np.uint32)
    b = np.asarray(points_with_c_['b'], dtype=np.uint32)
    rgb_colors = np.vstack([r, g, b]).T
    # rgb = rgb_colors.astype(np.float) / 255
    ind_good_points_ = np.sum(rgb_colors[:] < 210, axis=-1) == 3
    ind_good_points_ = np.where(ind_good_points_ == 1)[0]
    new_points_ = points_[ind_good_points_]
    return new_points_


def get_voxel_fun(points_, n):
    get_voxel = voxelgrid.VoxelGrid(points_, n_x=n, n_y=n, n_z=n)
    get_voxel.compute()
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


def cal_grasp(msg, cam_pos_):
    points_ = pointclouds.pointcloud2_to_xyz_array(msg)
    points_ = points_.astype(np.float32)
    remove_white = False
    if remove_white:
        points_ = remove_white_pixel(msg, points_, vis=True)
    # begin voxel points
    n = n_voxel  # parameter related to voxel method
    # gpg improvements, highlights: flexible n parameter for voxelizing.
    points_voxel_ = get_voxel_fun(points_, n)
    if len(points_) < 2000:  # should be a parameter
        while len(points_voxel_) < len(points_)-15:
            points_voxel_ = get_voxel_fun(points_, n)
            n = n + 100
            rospy.loginfo("the voxel has {} points, we want get {} points".format(len(points_voxel_), len(points_)))

    rospy.loginfo("the voxel has {} points, we want get {} points".format(len(points_voxel_), len(points_)))
    points_ = points_voxel_
    remove_points = False
    if remove_points:
        points_ = remove_table_points(points_, vis=True)
    point_cloud = pcl.PointCloud(points_)
    norm = point_cloud.make_NormalEstimation()
    norm.set_KSearch(30)  # critical parameter when calculating the norms
    normals = norm.compute()
    surface_normal = normals.to_array()
    surface_normal = surface_normal[:, 0:3]
    vector_p2cam = cam_pos_ - points_
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)
    tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    tmp = np.ones([len(angel), 3])
    tmp[wrong_dir_norm, :] = -1
    surface_normal = surface_normal * tmp
    select_point_above_table = 0.010
    #  modify of gpg: make it as a parameter. avoid select points near the table.
    points_for_sample = points_[np.where(points_[:, 2] > select_point_above_table)[0]]
    if len(points_for_sample) == 0:
        rospy.loginfo("Can not seltect point, maybe the point cloud is too low?")
        return [], points_, surface_normal
    yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
    if not using_mp:
        rospy.loginfo("Begin cal grasps using single thread, slow!")
        grasps_together_ = ags.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps,
                                             max_num_samples=max_num_samples, show_final_grasp=show_final_grasp)
    else:
        # begin parallel grasp:
        rospy.loginfo("Begin cal grasps using parallel!")

        def grasp_task(num_grasps_, ags_, queue_):
            ret = ags_.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_,
                                     max_num_samples=max_num_samples, show_final_grasp=show_final_grasp)
            queue_.put(ret)

        queue = mp.Queue()
        num_grasps_p_worker = int(num_grasps/num_workers)
        workers = [mp.Process(target=grasp_task, args=(num_grasps_p_worker, ags, queue)) for _ in range(num_workers)]
        [i.start() for i in workers]

        grasps_together_ = []
        for i in range(num_workers):
            grasps_together_ = grasps_together_ + queue.get()
        rospy.loginfo("Finish mp processing!")
    rospy.loginfo("Grasp sampler finish, generated {} grasps.".format(len(grasps_together_)))
    return grasps_together_, points_, surface_normal


def check_collision_square(grasp_bottom_center, approach_normal, binormal,
                           minor_pc, points_, p, way="p_open"):
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    matrix_ = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
    grasp_matrix = matrix_.T
    points_ = points_ - grasp_bottom_center.reshape(1, 3)
    tmp = np.dot(grasp_matrix, points_.T)
    points_g = tmp.T
    use_dataset_py = True
    if not use_dataset_py:
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
    # for the way of pointGPD/dataset.py:
    else:
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth
        z_limit = width / 4
        y_limit = width / 2
        x1 = points_g[:, 0] > 0
        x2 = points_g[:, 0] < x_limit
        y1 = points_g[:, 1] > -y_limit
        y2 = points_g[:, 1] < y_limit
        z1 = points_g[:, 2] > -z_limit
        z2 = points_g[:, 2] < z_limit
        a = np.vstack([x1, x2, y1, y2, z1, z2])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

    return has_p, points_in_area, points_g


def collect_pc(grasp_, pc):
    """
    grasp_bottom_center, normal, major_pc, minor_pc
    """
    grasp_num = len(grasp_)
    grasp_ = np.array(grasp_)
    grasp_ = grasp_.reshape(-1, 5, 3)  # prevent to have grasp that only have number 1
    grasp_bottom_center = grasp_[:, 0]
    approach_normal = grasp_[:, 1]
    binormal = grasp_[:, 2]
    minor_pc = grasp_[:, 3]

    in_ind_ = []
    in_ind_points_ = []
    p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    for i_ in range(grasp_num):
        has_p, in_ind_tmp, points_g = check_collision_square(grasp_bottom_center[i_], approach_normal[i_],
                                                             binormal[i_], minor_pc[i_], pc, p)
        in_ind_.append(in_ind_tmp)
        in_ind_points_.append(points_g[in_ind_[i_]])
    return in_ind_, in_ind_points_


def show_marker(marker_array_, pos_, ori_, scale_, color_, lifetime_):
    marker_ = Marker()
    marker_.header.frame_id = "/table_top"
    # marker_.header.stamp = rospy.Time.now()
    marker_.type = marker_.CUBE
    marker_.action = marker_.ADD

    marker_.pose.position.x = pos_[0]
    marker_.pose.position.y = pos_[1]
    marker_.pose.position.z = pos_[2]
    marker_.pose.orientation.x = ori_[1]
    marker_.pose.orientation.y = ori_[2]
    marker_.pose.orientation.z = ori_[3]
    marker_.pose.orientation.w = ori_[0]

    marker_.lifetime = rospy.Duration.from_sec(lifetime_)
    marker_.scale.x = scale_[0]
    marker_.scale.y = scale_[1]
    marker_.scale.z = scale_[2]
    marker_.color.a = 0.5
    red_, green_, blue_ = color_
    marker_.color.r = red_
    marker_.color.g = green_
    marker_.color.b = blue_
    marker_array_.markers.append(marker_)


def show_grasp_marker(marker_array_, real_grasp_, gripper_, color_, lifetime_):
    """
    show grasp using marker
    :param marker_array_: marker array
    :param real_grasp_: [0] position, [1] approach [2] binormal [3] minor pc
    :param gripper_: gripper parameter of a grasp
    :param color_: color of the gripper
    :param lifetime_: time for showing the maker
    :return: return add makers to the maker array

    """
    hh = gripper_.hand_height
    fw = gripper_.real_finger_width
    hod = gripper_.hand_outer_diameter
    hd = gripper_.real_hand_depth
    open_w = hod - fw * 2

    approach = real_grasp_[1]
    binormal = real_grasp_[2]
    minor_pc = real_grasp_[3]
    grasp_bottom_center = real_grasp_[4] - approach * (gripper_.real_hand_depth - gripper_.hand_depth)

    rotation = np.vstack([approach, binormal, minor_pc]).T
    qua = Quaternion(matrix=rotation)

    marker_bottom_pos = grasp_bottom_center - approach * hh * 0.5
    marker_left_pos = grasp_bottom_center - binormal * (open_w * 0.5 + fw * 0.5) + hd * 0.5 * approach
    marker_right_pos = grasp_bottom_center + binormal * (open_w * 0.5 + fw * 0.5) + hd * 0.5 * approach
    show_marker(marker_array_, marker_bottom_pos, qua, np.array([hh, hod, hh]), color_, lifetime_)
    show_marker(marker_array_, marker_left_pos, qua, np.array([hd, fw, hh]), color_, lifetime_)
    show_marker(marker_array_, marker_right_pos, qua, np.array([hd, fw, hh]), color_, lifetime_)


def check_hand_points_fun(real_grasp_):
    ind_points_num = []
    for i in range(len(real_grasp_)):
        grasp_bottom_center = real_grasp_[i][4]
        approach_normal = real_grasp_[i][1]
        binormal = real_grasp_[i][2]
        minor_pc = real_grasp_[i][3]
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    binormal, minor_pc, points,
                                                                    local_hand_points, "p_open")
        ind_points_num.append(len(ind_points_tmp))
    print(ind_points_num)
    file_name = "./generated_grasps/real_points/" + str(np.random.randint(300)) + str(len(real_grasp_)) + ".npy"
    np.save(file_name, np.array(ind_points_num))


def get_grasp_msg(real_good_grasp_, score_value_):
    grasp_bottom_center_modify = real_good_grasp_[4]
    approach = real_good_grasp_[1]
    binormal = real_good_grasp_[2]
    minor_pc = real_good_grasp_[3]
    grasp_config_ = GraspConfig()
    top_p_ = grasp_bottom_center_modify + approach * ags.gripper.hand_depth
    grasp_config_.sample.x = grasp_bottom_center_modify[0]
    grasp_config_.sample.y = grasp_bottom_center_modify[1]
    grasp_config_.sample.z = grasp_bottom_center_modify[2]
    grasp_config_.top.x = top_p_[0]
    grasp_config_.top.y = top_p_[1]
    grasp_config_.top.z = top_p_[2]
    grasp_config_.approach.x = approach[0]
    grasp_config_.approach.y = approach[1]
    grasp_config_.approach.z = approach[2]
    grasp_config_.binormal.x = binormal[0]
    grasp_config_.binormal.y = binormal[1]
    grasp_config_.binormal.z = binormal[2]
    grasp_config_.axis.x = minor_pc[0]
    grasp_config_.axis.y = minor_pc[1]
    grasp_config_.axis.z = minor_pc[2]
    grasp_config_.score.data = score_value_

    return grasp_config_


def remove_grasp_outside_tray(grasps_, points_):
    x_min = points_[:, 0].min()
    x_max = points_[:, 0].max()
    y_min = points_[:, 1].min()
    y_max = points_[:, 1].max()
    valid_grasp_ind_ = []
    for i in range(len(grasps_)):
        grasp_bottom_center = grasps_[i][4]
        approach_normal = grasps_[i][1]
        major_pc = grasps_[i][2]
        hand_points_ = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
        finger_points_ = hand_points_[[1, 2, 3, 4, 9, 10, 13, 14], :]
        # aa = points_[:, :2] - finger_points_[0][:2]  # todo： work of remove outside grasp not finished.

        a = finger_points_[:, 0] < x_min
        b = finger_points_[:, 0] > x_max
        c = finger_points_[:, 1] < y_min
        d = finger_points_[:, 1] > y_max
        if np.sum(a) + np.sum(b) + np.sum(c) + np.sum(d) == 0:
            valid_grasp_ind_.append(i)
    grasps_inside_ = [grasps_[i] for i in valid_grasp_ind_]
    rospy.loginfo("gpg got {} grasps, after remove grasp outside tray, {} grasps left".format(len(grasps_),
                                                                                              len(grasps_inside_)))
    return grasps_inside_


if __name__ == '__main__':
    """
    definition of gotten grasps:

    grasp_bottom_center = grasp_[0]
    approach_normal = grasp_[1]
    binormal = grasp_[2]
    """

    rospy.init_node('grasp_tf_broadcaster', anonymous=True)
    pub1 = rospy.Publisher('gripper_vis', MarkerArray, queue_size=1)
    pub2 = rospy.Publisher('/detect_grasps/clustered_grasps', GraspConfigList, queue_size=1)
    rate = rospy.Rate(10)
    rospy.set_param("/robot_at_home", "true")  # only use when in simulation test.
    rospy.loginfo("getting transform from kinect2 to table top")
    cam_pos = None
    if cam_pos is None:
        print("Please change the above line to the position between /table_top and /kinect2_ir_optical_frame")
        print("In ROS, you can run: rosrun tf tf_echo /table_top /kinect2_ir_optical_frame")
        exit()
    while not rospy.is_shutdown():
        if rospy.get_param("/robot_at_home") == "false":
            robot_at_home = False
        else:
            robot_at_home = True
        if not robot_at_home:
            rospy.loginfo("Robot is moving, waiting the robot go home.")
            continue
        else:
            rospy.loginfo("Robot is at home, safely catching point cloud data.")
            if single_obj_testing:
                input("Pleas put object on table and press any number to continue!")
        rospy.loginfo("rospy is waiting for message: /table_top_points")
        kinect_data = rospy.wait_for_message("/table_top_points", PointCloud2)
        real_good_grasp = []
        real_bad_grasp = []
        real_score_value = []

        repeat = 1  # speed up this try 10 time is too time consuming

        # begin of grasp detection
        # if there is no point cloud on table, waiting for point cloud.
        if kinect_data.data == '':
            rospy.loginfo("There is no points on the table, waiting...")
            continue
        real_grasp, points, normals_cal = cal_grasp(kinect_data, cam_pos)
        if tray_grasp:
            real_grasp = remove_grasp_outside_tray(real_grasp, points)

        check_grasp_points_num = True  # evaluate the number of points in a grasp
        check_hand_points_fun(real_grasp) if check_grasp_points_num else 0

        in_ind, in_ind_points = collect_pc(real_grasp, points)
        if save_grasp_related_file:
            np.save("./generated_grasps/points.npy", points)
            np.save("./generated_grasps/in_ind.npy", in_ind)
            np.save("./generated_grasps/real_grasp.npy", real_grasp)
            np.save("./generated_grasps/cal_norm.npy", normals_cal)
        score = []  # should be 0 or 1
        score_value = []  # should be float [0, 1]
        ind_good_grasp = []
        ind_bad_grasp = []
        rospy.loginfo("Begin send grasp into pointnet, cal grasp score")
        for ii in range(len(in_ind_points)):
            if rospy.get_param("/robot_at_home") == "false":
                robot_at_home = False
            else:
                robot_at_home = True
            if not robot_at_home:
                rospy.loginfo("robot is not at home, stop calculating the grasp score")
                break
            if in_ind_points[ii].shape[0] < minimal_points_send_to_point_net:
                rospy.loginfo("Mark as bad grasp! Only {} points, should be at least {} points.".format(
                              in_ind_points[ii].shape[0], minimal_points_send_to_point_net))
                score.append(0)
                score_value.append(0.0)
                if show_bad_grasp:
                    ind_bad_grasp.append(ii)
            else:
                predict = []
                grasp_score = []
                for _ in range(repeat):
                    if len(in_ind_points[ii]) >= input_points_num:
                        points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),
                                                                           input_points_num, replace=False)]
                    else:
                        points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),
                                                                           input_points_num, replace=True)]
                    if_good_grasp, grasp_score_tmp = test_network(model.eval(), points_modify)
                    predict.append(if_good_grasp.item())
                    grasp_score.append(grasp_score_tmp)

                predict_vote = mode(predict)[0][0]  # vote from all the "repeat" results.
                grasp_score = np.array(grasp_score)
                if args.model_type == "3class":  # the best in 3 class classification is the last column, third column
                    which_one_is_best = 2  # should set as 2
                else:  # for two class classification best is the second column (also the last column)
                    which_one_is_best = 1  # should set as 1
                score_vote = np.mean(grasp_score[np.where(predict == predict_vote)][:, 0, which_one_is_best])
                score.append(predict_vote)
                score_value.append(score_vote)

                if score[ii] == which_one_is_best:
                    ind_good_grasp.append(ii)
                else:
                    if show_bad_grasp:
                        ind_bad_grasp.append(ii)
        print("Got {} good grasps, and {} bad grasps".format(len(ind_good_grasp),
                                                             len(in_ind_points) - len(ind_good_grasp)))
        if len(ind_good_grasp) != 0:
            real_good_grasp = [real_grasp[i] for i in ind_good_grasp]
            real_score_value = [score_value[i] for i in ind_good_grasp]
            if show_bad_grasp:
                real_bad_grasp = [real_grasp[i] for i in ind_bad_grasp]
        # end of grasp detection
        # get sorted ind by the score values
        sorted_value_ind = list(index for index, item in sorted(enumerate(real_score_value),
                                                                key=lambda item: item[1],
                                                                reverse=True))
        # sort grasps using the ind
        sorted_real_good_grasp = [real_good_grasp[i] for i in sorted_value_ind]
        real_good_grasp = sorted_real_good_grasp
        # get the sorted score value, from high to low
        real_score_value = sorted(real_score_value, reverse=True)

        marker_array = MarkerArray()
        marker_array_single = MarkerArray()
        grasp_msg_list = GraspConfigList()

        for i in range(len(real_good_grasp)):
            grasp_msg = get_grasp_msg(real_good_grasp[i], real_score_value[i])
            grasp_msg_list.grasps.append(grasp_msg)
        for i in range(len(real_good_grasp)):
            show_grasp_marker(marker_array, real_good_grasp[i], gripper, (0, 1, 0), marker_life_time)

        if show_bad_grasp:
            for i in range(len(real_bad_grasp)):
                show_grasp_marker(marker_array, real_bad_grasp[i], gripper, (1, 0, 0), marker_life_time)

        id_ = 0
        for m in marker_array.markers:
            m.id = id_
            id_ += 1

        grasp_msg_list.header.stamp = rospy.Time.now()
        grasp_msg_list.header.frame_id = "/table_top"

        if len(real_good_grasp) != 0:
            i = 0
            single_grasp_list_pub = GraspConfigList()
            single_grasp_list_pub.header.stamp = rospy.Time.now()
            single_grasp_list_pub.header.frame_id = "/table_top"
            grasp_msg = get_grasp_msg(real_good_grasp[i], real_score_value[i])
            single_grasp_list_pub.grasps.append(grasp_msg)
            show_grasp_marker(marker_array_single, real_good_grasp[i], gripper, (1, 0, 0), marker_life_time+20)

            for m in marker_array_single.markers:
                m.id = id_
                id_ += 1
            pub1.publish(marker_array)
            rospy.sleep(4)
            pub2.publish(single_grasp_list_pub)
            pub1.publish(marker_array_single)
        # pub2.publish(grasp_msg_list)
        rospy.loginfo(" Publishing grasp pose to rviz using marker array and good grasp pose")
        rate.sleep()
