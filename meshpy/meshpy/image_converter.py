"""
Classes to convert binary images to extruded meshes 
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np
import os
from PIL import Image, ImageDraw
import sklearn.decomposition
import sys

import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.transform import resize

from autolab_core import RigidTransform
from meshpy import Mesh3D
from autolab_core import BinaryImage


class ImageToMeshConverter:
    """ Namespace class for converting binary images to SDFs and meshes. """
    
    @staticmethod
    def binary_image_to_mesh(binary_im, extrusion=1000, scale_factor=1.0):
        """
        Converts a binary image to a 3D extruded polygonal mesh
        
        Parameters
        ----------
        binary_im : :obj:`perception.BinaryImage`
            binary image for silhouette
        extrusion : float
            amount to extrude the polygon in meters
        scale_factor : float
            amount to rescale the final mesh (from units of pixels to meters)

        Returns
        -------
        :obj:`Mesh3D`
            the resulting mesh

        Raises
        ------
        :obj:`ValueError`
            if the triangulation was not successful due to topology or other factors
        """
        # check valid input
        if not isinstance(binary_im, BinaryImage):
            raise ValueError('Must provide perception.BinaryImage as input')

        # get occupied indices from binary image
        binary_data = binary_im.data
        occ_coords = binary_im.nonzero_pixels()

        # create mesh faces and concatenate
        front_face_depth = extrusion / 2.0
        back_face_depth = -extrusion / 2.0
        front_verts, front_tris, front_ind_map = ImageToMeshConverter.create_mesh_face(occ_coords, front_face_depth,
                                                                                       binary_data.shape, cw=True)
        back_verts, back_tris, back_ind_map = ImageToMeshConverter.create_mesh_face(occ_coords, back_face_depth,
                                                                                    binary_data.shape, cw=False)
        verts, tris = ImageToMeshConverter.join_vert_tri_lists(front_verts, front_tris, back_verts, back_tris)
        num_verts = len(front_verts)
        back_ind_map = back_ind_map + num_verts

        # connect boundaries
        boundary_im = binary_im.boundary_map()
        ImageToMeshConverter.add_boundary_tris(boundary_im, verts, tris, front_ind_map, back_ind_map)

        # convert to mesh and clean
        m = Mesh3D(verts, tris)
        m.remove_unreferenced_vertices()
        T_im_world = RigidTransform(rotation=np.array([[0, 1, 0],
                                                       [-1, 0, 0],
                                                       [0, 0, 1]]),
                                    from_frame='obj',
                                    to_frame='obj')
        m = m.transform(T_im_world)
        m.rescale_dimension(scale_factor, Mesh3D.ScalingTypeRelative)
        return m

    @staticmethod
    def join_vert_tri_lists(verts1, tris1, verts2, tris2):
        """
        Concatenates two lists of vertices and triangles.
        
        Parameters
        ----------
        verts1 : :obj:`list` of 3-:obj:`list` of float
            first list of vertices
        tris1 : :obj:`list` of 3-:obj`list` of int
            first list of triangles
        verts2 : :obj:`list` of 3-:obj:`list` of float
            second list of vertices
        tris2 : :obj:`list` of 3-:obj`list` of int
            second list of triangles

        Returns
        -------
        verts : :obj:`list` of 3-:obj:`list` of float
            joined list of vertices
        tris : :obj:`list` of 3-:obj`list` of int
            joined list of triangles
        """
        num_verts1 = len(verts1)

        # simple append for verts
        verts = list(verts1)
        verts.extend(verts2)

        # offset and append triangle (vertex indices)
        tris = list(tris1)
        tris2_offset = [[num_verts1 + t[0], num_verts1 + t[1], num_verts1 + t[2]] for t in tris2]
        tris.extend(tris2_offset)
        return verts, tris

    @staticmethod
    def add_boundary_tris(boundary_im, verts, tris, front_ind_map, back_ind_map):
        """
        Connects front and back faces along the boundary, modifying tris IN PLACE
        NOTE: Right now this only works for points topologically equivalent to a sphere, eg. no holes! 
        This can be extended by parsing back over untriangulated boundary points.

        Parameters
        ----------
        boundary_im : :obj:`perception.BinaryImage`
            binary image of the boundary
        verts : :obj:`list` of 3-:obj:`list` of float
            list of vertices
        tris : :obj:`list` of 3-:obj`list` of int
            list of triangles
        front_ind_map : :obj:`numpy.ndarray`
            maps vertex coords to the indices of their front face vertex in list  
        back_ind_map : :obj:`numpy.ndarray`
            maps vertex coords to the indices of their back face vertex in list  

        Raises
        ------
        :obj:`ValueError`
            triangulation failed
        """
        # TODO: fix multiple connected comps

        # setup variables for boundary coords
        upper_bound = np.iinfo(np.uint8).max
        remaining_boundary = boundary_im.data.copy()
        boundary_ind = np.where(remaining_boundary == upper_bound)
        boundary_coords = list(zip(boundary_ind[0], boundary_ind[1]))
        if len(boundary_coords) == 0:
            raise ValueError('No boundary coordinates')

        # setup inital vars
        tris_arr = np.array(tris)
        visited_map = np.zeros(boundary_im.shape)
        another_visit_avail = True

        # make sure to start with a reffed tri
        visited_marker = 128
        finished = False
        it = 0
        i = 0
        coord_visits = []

        while not finished:
            finished = True
            logging.info('Boundary triangulation iter %d' %(it))
            reffed = False
            while not reffed and i < len(boundary_coords):
                cur_coord = boundary_coords[i]
                if visited_map[cur_coord[0], cur_coord[1]] == 0:
                    visited_map[cur_coord[0], cur_coord[1]] = 1
                    front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
                    back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
                    ref_tris = np.where(tris_arr == front_ind)
                    ref_tris = ref_tris[0]
                    reffed = (ref_tris.shape[0] > 0)
                    remaining_boundary[cur_coord[0], cur_coord[1]] = visited_marker
                i = i+1

            coord_visits.extend([cur_coord])
            cur_dir_angle = np.pi / 2 # start straight down

            # loop around boundary and add faces connecting front and back
            while another_visit_avail:
                front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
                back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
                ref_tris = np.where(tris_arr == front_ind)
                ref_tris = ref_tris[0]
                num_reffing_tris = ref_tris.shape[0]
                
                # get all possible cadidates from neighboring tris
                another_visit_avail = False
                candidate_next_coords = []
                for i in range(num_reffing_tris):
                    reffing_tri = tris[ref_tris[i]]
                    for j in range(3):
                        v = verts[reffing_tri[j]]
                        if boundary_im[v[0], v[1]] == upper_bound and visited_map[v[0], v[1]] == 0:
                            candidate_next_coords.append([v[0], v[1]])
                            another_visit_avail = True

                # get the "rightmost" next point
                num_candidates = len(candidate_next_coords)
                if num_candidates > 0:
                    # calculate candidate directions
                    directions = []
                    next_dirs = np.array(candidate_next_coords) - np.array(cur_coord)
                    dir_norms = np.linalg.norm(next_dirs, axis = 1)
                    next_dirs = next_dirs / np.tile(dir_norms, [2, 1]).T
                    
                    # calculate angles relative to positive x axis
                    new_angles = np.arctan(next_dirs[:,0] / next_dirs[:,1])
                    negative_ind = np.where(next_dirs[:,1] < 0)
                    negative_ind = negative_ind[0]
                    new_angles[negative_ind] = new_angles[negative_ind] + np.pi

                    # compute difference in angles
                    angle_diff = new_angles - cur_dir_angle
                    correction_ind = np.where(angle_diff <= -np.pi)
                    correction_ind = correction_ind[0]
                    angle_diff[correction_ind] = angle_diff[correction_ind] + 2 * np.pi
                    
                    # choose the next coordinate with the maximum angle diff (rightmost)
                    next_ind = np.where(angle_diff == np.max(angle_diff))
                    next_ind = next_ind[0]

                    cur_coord = candidate_next_coords[next_ind[0]]
                    cur_dir_angle = new_angles[next_ind[0]]

                    # add triangles (only add if there is a new candidate)
                    next_front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
                    next_back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
                    tris.append([int(front_ind), int(back_ind), int(next_front_ind)])
                    tris.append([int(back_ind), int(next_back_ind), int(next_front_ind)])

                    # mark coordinate as visited
                    visited_map[cur_coord[0], cur_coord[1]] = 1
                    coord_visits.append(cur_coord)
                    remaining_boundary[cur_coord[0], cur_coord[1]] = visited_marker

            # add edge back to first coord
            cur_coord = coord_visits[0]
            next_front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
            next_back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
            tris.append([int(front_ind), int(back_ind), int(next_front_ind)])
            tris.append([int(back_ind), int(next_back_ind), int(next_front_ind)])

            # check success 
            finished = (np.sum(remaining_boundary == upper_bound) == 0) or (i == len(boundary_coords))
            it += 1

    @staticmethod
    def create_mesh_face(occ_coords, depth, index_shape, cw=True):
        """
        Creates a 2D mesh face of vertices and triangles from the given coordinates at a specified depth.
        
        Parameters
        ----------
        occ_coords : :obj:`list` of 3-:obj:`tuple
            the coordinates of vertices
        depth : float
            the depth at which to place the face
        index_shape : 2-:obj:`tuple`
            the shape of the numpy grid on which the vertices lie
        cw : bool
            clockwise or counterclockwise orientation

        Returns
        -------
        verts : :obj:`list` of 3-:obj:`list` of float
            list of vertices
        tris : :obj:`list` of 3-:obj`list` of int
            list of triangles
        """
        # get mesh vertices
        verts = []
        tris = []
        ind_map = -1 * np.ones(index_shape) # map vertices to indices in vert list
        for coord in occ_coords:
            verts.append([coord[0], coord[1], depth])
            ind_map[coord[0], coord[1]] = len(verts) - 1

        # get mesh triangles
        # rule: vertex adds triangles that it is the 90 degree corner of
        for coord in occ_coords:
            coord_right = [coord[0] + 1, coord[1]]
            coord_left  = [coord[0] - 1, coord[1]]
            coord_below = [coord[0], coord[1] + 1]
            coord_above = [coord[0], coord[1] - 1]
            cur_ind = ind_map[coord[0], coord[1]]

            # add tri above left
            if coord_left[0] >= 0 and coord_above[1] >= 0:
                left_ind = ind_map[coord_left[0], coord_left[1]]
                above_ind = ind_map[coord_above[0], coord_above[1]]

                # check if valid vertices and add
                if left_ind > -1 and above_ind > -1:
                    if cw:
                        tris.append([int(cur_ind), int(left_ind), int(above_ind)])
                    else:
                        tris.append([int(cur_ind), int(above_ind), int(left_ind)])                        
                elif above_ind > -1:
                    # try to patch area
                    coord_left_above = [coord[0] - 1, coord[1] - 1]
                    if coord_left_above[0] > 0 and coord_left_above[1] > 0:
                        left_above_ind = ind_map[coord_left_above[0], coord_left_above[1]]

                        # check validity
                        if left_above_ind > -1:
                            if cw:
                                tris.append([int(cur_ind), int(left_above_ind), int(above_ind)])
                            else:
                                tris.append([int(cur_ind), int(above_ind), int(left_above_ind)])                                

            # add tri below right
            if coord_right[0] < index_shape[1] and coord_below[1] < index_shape[0]:
                right_ind = ind_map[coord_right[0], coord_right[1]]
                below_ind = ind_map[coord_below[0], coord_below[1]]

                # check if valid vertices and add
                if right_ind > -1 and below_ind > -1:
                    if cw:
                        tris.append([int(cur_ind), int(right_ind), int(below_ind)])
                    else:
                        tris.append([int(cur_ind), int(below_ind), int(right_ind)])
                elif below_ind > -1:
                    # try to patch area
                    coord_right_below = [coord[0] + 1, coord[1] + 1]
                    if coord_right_below[0] < index_shape[0] and coord_right_below[1] < index_shape[1]:
                        right_below_ind = ind_map[coord_right_below[0], coord_right_below[1]]

                        # check validity
                        if right_below_ind > -1:
                            if cw:
                                tris.append([int(cur_ind), int(right_below_ind), int(below_ind)])
                            else:
                                tris.append([int(cur_ind), int(below_ind), int(right_below_ind)])

        return verts, tris, ind_map

