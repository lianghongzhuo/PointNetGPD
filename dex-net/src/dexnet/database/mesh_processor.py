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
Encapsulates mesh cleaning & preprocessing pipeline for database generation
Authors: Mel Roderick and Jeff Mahler
"""
import glob
import IPython
import logging
import numpy as np
import os
import sklearn.decomposition

import meshpy.obj_file as obj_file
import meshpy.stp_file as stp_file
import meshpy.sdf_file as sdf_file
import xml.etree.cElementTree as et

from dexnet.constants import *

class RescalingType:
    """
    Enum to specify different rules for rescaling meshes
    """
    FIT_MIN_DIM = 'min'
    FIT_MED_DIM = 'med'
    FIT_MAX_DIM = 'max'
    FIT_DIAG = 'diag'
    RELATIVE = 'relative'

class MeshProcessor:
    """
    Preprocessing of mesh files into graspable objects for planning with Dex-Net.

    Parameters
    ----------
    filename : :obj:`str`
        name of the mesh file to process
    cache_dir : :obj:`str`
        directory to store intermediate files to
    """

    def __init__(self, filename, cache_dir):
        file_path, file_root = os.path.split(filename)
        file_root, file_ext = os.path.splitext(file_root)
        self.file_path_ = file_path
        self.file_root_ = file_root
        self.file_ext_ = file_ext
        self.cache_dir_ = cache_dir
        self.key_ = file_root

    @property
    def file_path(self):
        return self.file_path_

    @property
    def file_root(self):
        return self.file_root_

    @property
    def key(self):
        return self.key_

    @property
    def file_ext(self):
        return self.file_ext_

    @property
    def cache_dir(self):
        return self.cache_dir_

    @property
    def filename(self):
        return os.path.join(self.file_path, self.file_root + self.file_ext)

    @property
    def mesh(self):
        return self.mesh_

    @property
    def sdf(self):
        return self.sdf_

    @property
    def stable_poses(self):
        return self.stable_poses_

    @property
    def orig_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + self.file_ext_)

    @property
    def obj_filename(self):
        return os.path.join(self.cache_dir_, self.file_root_ + PROC_TAG + OBJ_EXT)

    @property
    def off_filename(self):
        return os.path.join(self.cache_dir_, self.file_root_ + PROC_TAG + OFF_EXT)

    @property
    def sdf_filename(self):
        return os.path.join(self.cache_dir_, self.file_root_ + PROC_TAG + SDF_EXT)

    @property
    def stp_filename(self):
        return os.path.join(self.cache_dir_, self.file_root_ + PROC_TAG + STP_EXT)

    def generate_graspable(self, config):
        """ Generates a graspable object based on the given configuration.

        Parameters
        ----------
        config : :obj:`dict`
            dictionary containing values for preprocessing parameters (preprocessing meshlab script, object density, object scale, object rescaling type, path to the SDFGen binary, the dimension of the sdf grid, the amount of sdf padding to use, and the min probability of stable poses to prune)

        Notes
        -----
        Required configuration key-value pairs in Other Parameters.

        Other Parameters
        ----------------
        obj_density : float
            density of object
        obj_scale : float
            scale of object
        path_to_sdfgen : :obj:`str`
            path to the SDFGen binary
        sdf_dim : int
            dimensions of signed distance field grid
        sdf_padding : int
            how much to pad the boundary of the sdf grid
        stp_min_prob : float
            minimum probability for stored stable poses
        """
        preproc_script = None
        if 'preproc_script' in list(config.keys()):
            preproc_script = config['preproc_script']
        self._load_mesh(preproc_script)
        self.mesh_.density = config['obj_density']
        self._clean_mesh(config['obj_target_scale'], config['obj_scaling_mode'], config['use_uniform_com'], rescale_mesh=config['rescale_objects'])
        self._generate_sdf(config['path_to_sdfgen'], config['sdf_dim'], config['sdf_padding'])
        self._generate_stable_poses(config['stp_min_prob'])
        return self.mesh, self.sdf, self.stable_poses
        
    def _load_mesh(self, script_to_apply=None):
        """ Loads the mesh from the file by first converting to an obj and then loading """        
        # convert to an obj file using meshlab
        if script_to_apply is None:
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(self.filename, self.obj_filename)
        else:
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\" -s \"%s\"' %(self.filename, self.obj_filename, script_to_apply) 
        os.system(meshlabserver_cmd)
        logging.info('MeshlabServer Command: %s' %(meshlabserver_cmd))

        if not os.path.exists(self.obj_filename):
            raise ValueError('Meshlab conversion failed for %s' %(self.obj_filename))
        
        # read mesh from obj file
        of = obj_file.ObjFile(self.obj_filename)
        self.mesh_ = of.read()
        return self.mesh_ 

    def _clean_mesh(self, scale, rescaling_type, use_uniform_com, rescale_mesh=False):
        """ Runs all cleaning ops at once """
        self._remove_bad_tris()
        self._remove_unreferenced_vertices()
        self._standardize_pose()
        if rescale_mesh:
            self._rescale_vertices(scale, rescaling_type, use_uniform_com)

    def _remove_bad_tris(self):
        """ Remove triangles with illegal out-of-bounds references """
        new_tris = []
        num_v = len(self.mesh_.vertices)
        for t in self.mesh_.triangles.tolist():
            if (t[0] >= 0 and t[0] < num_v and t[1] >= 0 and t[1] < num_v and t[2] >= 0 and t[2] < num_v and
                t[0] != t[1] and t[0] != t[2] and t[1] != t[2]):
                new_tris.append(t)
        self.mesh_.triangles = new_tris
        return self.mesh_

    def _remove_unreferenced_vertices(self):
        """ Clean out vertices (and normals) not referenced by any triangles. """
        # convert vertices to an array
        vertex_array = np.array(self.mesh_.vertices)
        num_v = vertex_array.shape[0]

        # fill in a 1 for each referenced vertex
        reffed_array = np.zeros([num_v, 1])
        for f in self.mesh_.triangles.tolist():
            if f[0] < num_v and f[1] < num_v and f[2] < num_v:
                reffed_array[f[0]] = 1
                reffed_array[f[1]] = 1
                reffed_array[f[2]] = 1

        # trim out vertices that are not referenced
        reffed_v_old_ind = np.where(reffed_array == 1)
        reffed_v_old_ind = reffed_v_old_ind[0]
        reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1 # counts number of reffed v before each ind

        try:
            self.mesh_.vertices = vertex_array[reffed_v_old_ind, :]
            if self.mesh_.normals is not None:
                normals_array = np.array(self.mesh_.normals)
                self.mesh_.set_normals(normals_array[reffed_v_old_ind, :].tolist())
        except IndexError:
            return False

        # create new face indices
        new_triangles = []
        for f in self.mesh_.triangles:
            new_triangles.append([reffed_v_new_ind[f[0]], reffed_v_new_ind[f[1]], reffed_v_new_ind[f[2]]] )
        self.mesh_.triangles = new_triangles
        return True

    def _standardize_pose(self):
        """
        Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
        centroid and the principal axes are aligned with the vertical Z, Y, and X axes.
        
        Returns:
        Nothing. Modified the mesh in place (for now)
        """
        self.mesh_.center_vertices_bb()
        vertex_array_cent = np.array(self.mesh_.vertices)

        # find principal axes
        pca = sklearn.decomposition.PCA(n_components = 3)
        pca.fit(vertex_array_cent)

        # count num vertices on side of origin wrt principal axes
        comp_array = pca.components_
        norm_proj = vertex_array_cent.dot(comp_array.T)
        opposite_aligned = np.sum(norm_proj < 0, axis = 0)
        same_aligned = np.sum(norm_proj >= 0, axis = 0)

        # create rotation from principal axes to standard basis
        z_axis = comp_array[0,:]
        y_axis = comp_array[1,:]
        if opposite_aligned[2] > same_aligned[2]:
            z_axis = -z_axis
        if opposite_aligned[1] > same_aligned[1]:
            y_axis = -y_axis
        x_axis = np.cross(y_axis, z_axis)
        R_pc_obj = np.c_[x_axis, y_axis, z_axis]
        
        # rotate vertices, normals and reassign to the mesh
        vertex_array_rot = R_pc_obj.T.dot(vertex_array_cent.T)
        vertex_array_rot = vertex_array_rot.T
        self.mesh_.vertices = vertex_array_rot
        self.mesh_.center_vertices_bb()

        if self.mesh_.normals is not None:
            normals_array = np.array(self.mesh_.normals_)
            normals_array_rot = R_pc_obj.dot(normals_array.T)
            self.mesh_.set_normals(normals_array_rot.tolist())

    def _rescale_vertices(self, scale, rescaling_type=RescalingType.FIT_MIN_DIM, use_uniform_com=False):
        """
        Rescales the vertex coordinates so that the minimum dimension (X, Y, Z) is exactly min_scale
        
        Params:
        scale: (float) scale of the mesh
        rescaling_type: (int) which dimension to scale along; if not absolute then the min,med,max dim is scaled to be exactly scale
        Returns:
        Nothing. Modified the mesh in place (for now)
        """
        vertex_array = np.array(self.mesh_.vertices)
        min_vertex_coords = np.min(self.mesh_.vertices, axis=0)
        max_vertex_coords = np.max(self.mesh_.vertices, axis=0)
        vertex_extent = max_vertex_coords - min_vertex_coords

        # find minimal dimension
        if rescaling_type == RescalingType.FIT_MIN_DIM:
            dim = np.where(vertex_extent == np.min(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == RescalingType.FIT_MED_DIM:
            dim = np.where(vertex_extent == np.median(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == RescalingType.FIT_MAX_DIM:
            dim = np.where(vertex_extent == np.max(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == RescalingType.RELATIVE:
            relative_scale = 1.0
        elif rescaling_type == RescalingType.FIT_DIAG:
            diag = np.linalg.norm(vertex_extent)
            relative_scale = diag / 3.0 # make the gripper size exactly one third of the diagonal

        # compute scale factor and rescale vertices
        scale_factor = scale / relative_scale 
        vertex_array = scale_factor * vertex_array
        self.mesh_.vertices_ = vertex_array
        self.mesh_._compute_bb_center()
        self.mesh_._compute_centroid()
        self.mesh_.center_of_mass = self.mesh_.bb_center_
        if use_uniform_com:
            self.mesh_.center_of_mass = self.mesh_._compute_com_uniform()
        
    def _generate_sdf(self, path_to_sdfgen, dim, padding):
        """ Converts mesh to an sdf object """
        # write the mesh to file
        of = obj_file.ObjFile(self.obj_filename)
        of.write(self.mesh_)

        # create the SDF using binary tools
        sdfgen_cmd = '%s \"%s\" %d %d' %(path_to_sdfgen, self.obj_filename, dim, padding)
        os.system(sdfgen_cmd)
        logging.info('SDF Command: %s' %(sdfgen_cmd))

        if not os.path.exists(self.sdf_filename):
            raise ValueError('SDF computation failed for %s' %(self.sdf_filename))
        os.system('chmod a+rwx \"%s\"' %(self.sdf_filename) )

        # read the generated sdf
        sf = sdf_file.SdfFile(self.sdf_filename)
        self.sdf_ = sf.read()
        return self.sdf_

    def _generate_stable_poses(self, min_prob = 0.05):
        """ Computes mesh stable poses """
        self.stable_poses_ = self.mesh_.stable_poses(min_prob=min_prob)
        return self.stable_poses_

