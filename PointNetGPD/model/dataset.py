import os
import glob
import pickle
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np


class PointGraspDataset(torch.utils.data.Dataset):
    def __init__(self, obj_points_num, grasp_points_num, pc_file_used_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.obj_points_num = obj_points_num
        self.grasp_points_num = grasp_points_num
        self.pc_file_used_num = pc_file_used_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num = 50
        self.projection_margin = 1

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', '*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6]  # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis / np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width * axis / 2
        right = center + width * axis / 2
        # bottom = center - width*approach
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        # bottom = (transform @ np.array([bottom[0], bottom[1], bottom[2], 1]))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[
                       :3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(matrix, (pc - center).T)).T
        left_t = (-width * np.array([0, 1, 0]) / 2).squeeze()
        right_t = (width * np.array([0, 1, 0]) / 2).squeeze()

        x_limit = width / 4
        z_limit = width / 4
        y_limit = width / 2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                         [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic - margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1) / number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert (occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check) != 0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert (np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))
        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        fl_pc = fl_pc[np.random.choice(len(fl_pc), size=self.pc_file_used_num)]

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.vstack([np.load(i) for i in fl_pc])
        pc = pc[np.random.choice(len(pc), size=self.obj_points_num)]
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score * 0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 1
        else:
            return None

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount


class PointGraspMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, obj_points_num, grasp_points_num, pc_file_used_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.obj_points_num = obj_points_num
        self.grasp_points_num = grasp_points_num
        self.pc_file_used_num = pc_file_used_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num = 50
        self.projection_margin = 1

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', '*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6]  # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis / np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width * axis / 2
        right = center + width * axis / 2
        # bottom = center - width*approach
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        # bottom = (transform @ np.array([bottom[0], bottom[1], bottom[2], 1]))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[
                       :3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(matrix, (pc - center).T)).T
        left_t = (-width * np.array([0, 1, 0]) / 2).squeeze()
        right_t = (width * np.array([0, 1, 0]) / 2).squeeze()

        x_limit = width / 4
        z_limit = width / 4
        y_limit = width / 2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                         [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic - margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1) / number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert (occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check) != 0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert (np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))
        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        fl_pc = fl_pc[np.random.choice(len(fl_pc), size=self.pc_file_used_num)]

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.vstack([np.load(i) for i in fl_pc])
        pc = pc[np.random.choice(len(pc), size=self.obj_points_num)]
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score * 0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 2
        else:
            label = 1

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount


class PointGraspOneViewDataset(torch.utils.data.Dataset):
    def __init__(self, grasp_points_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.grasp_points_num = grasp_points_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num = 50
        self.projection_margin = 1
        self.minimum_point_amount = 150

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', 'pc_NP3_NP5*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]
        for k in self.d_pc.keys():
            self.d_pc[k].sort()

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6]  # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis / np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width * axis / 2
        right = center + width * axis / 2
        # bottom = center - width*approach
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        # bottom = (transform @ np.array([bottom[0], bottom[1], bottom[2], 1]))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[
                       :3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(matrix, (pc - center).T)).T
        left_t = (-width * np.array([0, 1, 0]) / 2).squeeze()
        right_t = (width * np.array([0, 1, 0]) / 2).squeeze()

        x_limit = width / 4
        z_limit = width / 4
        y_limit = width / 2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                         [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic - margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1) / number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert (occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check) != 0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert (np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))

        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        np.random.shuffle(fl_pc)

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.load(fl_pc[-1])
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score * 0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 1
        else:
            return None

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount


class PointGraspOneViewMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, grasp_points_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.grasp_points_num = grasp_points_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num = 50
        self.projection_margin = 1
        self.minimum_point_amount = 150

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', 'pc_NP3_NP5*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]
        for k in self.d_pc.keys():
            self.d_pc[k].sort()

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6]  # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis / np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width * axis / 2
        right = center + width * axis / 2
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[
                       :3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        pc_t = (np.dot(matrix, (pc - center).T)).T
        left_t = (-width * np.array([0, 1, 0]) / 2).squeeze()
        right_t = (width * np.array([0, 1, 0]) / 2).squeeze()

        x_limit = width / 4
        z_limit = width / 4
        y_limit = width / 2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                         [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic - margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1) / number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert (occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check) != 0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert (np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))

        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        np.random.shuffle(fl_pc)

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.load(fl_pc[-1])
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score * 0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 2
        else:
            label = 1

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount


def test_dataset():
    grasp_points_num = 1000
    obj_points_num = 50000
    pc_file_used_num = 20
    thresh_good = 0.6
    thresh_bad = 0.6

    input_size = 60
    input_chann = 12  # 12
    a = PointGraspDataset(
        obj_points_num=obj_points_num,
        grasp_points_num=grasp_points_num,
        pc_file_used_num=pc_file_used_num,
        path="../data",
        tag='train',
        grasp_amount_per_file=6500,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        projection=True,
        project_chann=input_chann,
        project_size=input_size,
    )
    c, d = a.__getitem__(0)


if __name__ == '__main__':
    test_dataset()
