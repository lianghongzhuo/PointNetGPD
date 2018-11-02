#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 08/09/2018 12:00 AM
# File Name  : get_ur5_robot_state.py
import rospy
import numpy as np
import moveit_commander

def get_robot_state_moveit():
    # moveit_commander.roscpp_initialize(sys.argv)
    # robot = moveit_commander.RobotCommander()
    try:
        current_joint_values = np.array(group.get_current_joint_values())
        diff = abs(current_joint_values - home_joint_values)*180/np.pi
        if np.sum(diff<1) == 6:  # if current joint - home position < 1 degree, we think it is at home
            return 1  # robot at home
        else:
            return 2  # robot is moving
    except:
        return 3  # robot state unknow
        rospy.loginfo("Get robot state failed")

if __name__ == '__main__':
    rospy.init_node('ur5_state_checker_if_it_at_home', anonymous=True)
    rate = rospy.Rate(10)
    group = moveit_commander.MoveGroupCommander("arm")
    home_joint_values = np.array([0, -1.5708, 0, -1.5708, 0, 0])
    while not rospy.is_shutdown():
        at_home = get_robot_state_moveit()
        if at_home == 1:
            rospy.set_param("/robot_at_home", "true")
            rospy.loginfo("robot at home")
        elif at_home == 2:
            rospy.set_param("/robot_at_home", "false")
            rospy.loginfo("robot is moving")
        elif at_home == 3:
            rospy.loginfo("robot state unknow")
        rate.sleep()
