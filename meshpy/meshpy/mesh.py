"""
Encapsulates mesh for grasping operations
Authors: Jeff Mahler and Matt Matl
"""
import math
try:
    import queue
except ImportError:
    import Queue as queue
import os
import random
from subprocess import Popen
import sys

import numpy as np
import scipy.spatial as ss
import sklearn.decomposition
import trimesh as tm

from autolab_core import RigidTransform, Point, Direction, PointCloud, NormalCloud

from . import obj_file
from . import stable_pose as sp


class Mesh3D(object):
    """A triangular mesh for a three-dimensional shape representation.

    Attributes
    ----------
    vertices : :obj:`numpy.ndarray` of float
        A #verts by 3 array, where each row contains an ordered
        [x,y,z] set that describes one vertex.
    triangles : :obj:`numpy.ndarray`  of int
        A #tris by 3 array, where each row contains indices of vertices in
        the `vertices` array that are part of the triangle.
    normals : :obj:`numpy.ndarray` of float
        A #normals by 3 array, where each row contains a normalized
        vector. This list should contain one norm per vertex.
    density : float
        The density of the mesh.
    center_of_mass : :obj:`numpy.ndarray` of float
        The 3D location of the mesh's center of mass.
    mass : float
        The mass of the mesh (read-only).
    inertia : :obj:`numpy.ndarray` of float
        The 3x3 inertial matrix of the mesh (read-only).
    bb_center : :obj:`numpy.ndarray` of float
        The 3D location of the center of the mesh's minimal bounding box
        (read-only).
    centroid : :obj:`numpy.ndarray` of float
        The 3D location of the mesh's vertex mean (read-only).
    """

    ScalingTypeMin = 0
    ScalingTypeMed = 1
    ScalingTypeMax = 2
    ScalingTypeRelative = 3
    ScalingTypeDiag = 4
    OBJ_EXT = '.obj'
    PROC_TAG = '_proc'
    C_canonical = np.array([[1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0]])

    def __init__(self, vertices, triangles, normals=None,
                 density=1.0, center_of_mass=None,
                 trimesh=None, T_obj_world=RigidTransform(from_frame='obj', to_frame='world')):
        """Construct a 3D triangular mesh.

        Parameters
        ----------
        vertices : :obj:`numpy.ndarray` of float
            A #verts by 3 array, where each row contains an ordered
            [x,y,z] set that describes one vertex.
        triangles : :obj:`numpy.ndarray`  of int
            A #tris by 3 array, where each row contains indices of vertices in
            the `vertices` array that are part of the triangle.
        normals : :obj:`numpy.ndarray` of float
            A #normals by 3 array, where each row contains a normalized
            vector. This list should contain one norm per vertex.
        density : float
            The density of the mesh.
        center_of_mass : :obj:`numpy.ndarray` of float
            The 3D location of the mesh's center of mass.
        uniform_com : bool
            Whether or not to assume a uniform mass density for center of mass comp
        """
        if vertices is not None:
            vertices = np.array(vertices)
        self.vertices_ = vertices

        if triangles is not None:
            triangles = np.array(triangles)
        self.triangles_ = triangles

        if normals is not None:
            normals = np.array(normals)
            if normals.shape[0] == 3:
                normals = normals.T
        self.normals_ = normals

        self.density_ = density

        self.center_of_mass_ = center_of_mass

        # Read-Only parameter initialization
        self.mass_ = None
        self.inertia_ = None
        self.bb_center_ = self._compute_bb_center()
        self.centroid_ = self._compute_centroid()
        self.surface_area_ = None
        self.face_dag_ = None
        self.trimesh_ = trimesh
        self.T_obj_world_ = T_obj_world

        if self.center_of_mass_ is None:
            if self.is_watertight:
                self.center_of_mass_ = np.array(self._compute_com_uniform())
            else:
                self.center_of_mass_ = np.array(self.bb_center_)

    ##################################################################
    # Properties
    ##################################################################

    # =============================================
    # Read-Write Properties
    # =============================================
    @property
    def vertices(self):
        """:obj:`numpy.ndarray` of float : A #verts by 3 array,
        where each row contains an ordered
        [x,y,z] set that describes one vertex.
        """
        return self.vertices_

    @vertices.setter
    def vertices(self, v):
        self.vertices_ = np.array(v)
        self.mass_ = None
        self.inertia_ = None
        self.normals_ = None
        self.surface_area_ = None
        self.bb_center_ = self._compute_bb_center()
        self.centroid_ = self._compute_centroid()

    @property
    def triangles(self):
        """:obj:`numpy.ndarray` of int : A #tris by 3 array,
        where each row contains indices of vertices in
        the `vertices` array that are part of the triangle.
        """
        return self.triangles_

    @triangles.setter
    def triangles(self, t):
        self.triangles_ = np.array(t)
        self.mass_ = None
        self.inertia_ = None
        self.surface_area_ = None

    @property
    def normals(self):
        """:obj:`numpy.ndarray` of float :
        A #normals by 3 array, where each row contains a normalized
        vector. This list should contain one norm per vertex.
        """
        return self.normals_

    @normals.setter
    def normals(self, n):
        self.normals_ = np.array(n)

    @property
    def density(self):
        """float : The density of the mesh.
        """
        return self.density_

    @density.setter
    def density(self, d):
        self.density_ = d
        self.mass_ = None
        self.inertia_ = None

    @property
    def center_of_mass(self):
        """:obj:`numpy.ndarray` of float :
        The 3D location of the mesh's center of mass.
        """
        return self.center_of_mass_

    @center_of_mass.setter
    def center_of_mass(self, com):
        self.center_of_mass_ = com
        self.inertia_ = None

    @property
    def num_vertices(self):
        """ :obj:`int`:
        The number of total vertices
        """
        return self.vertices.shape[0]

    @property
    def num_triangles(self):
        """ :obj:`int`:
        The number of total triangles
        """
        return self.triangles.shape[0]

    # =============================================
    # Read-Only Properties
    # =============================================
    @property
    def mass(self):
        """float : The mass of the mesh (read-only).
        """
        if self.mass_ is None:
            self.mass_ = self._compute_mass()
        return self.mass_

    @property
    def inertia(self):
        """:obj:`numpy.ndarray` of float :
        The 3x3 inertial matrix of the mesh (read-only).
        """
        if self.inertia_ is None:
            self.inertia_ = self._compute_inertia()
        return self.inertia_

    @property
    def bb_center(self):
        """:obj:`numpy.ndarray` of float :
        The 3D location of the center of the mesh's minimal bounding box
        (read-only).
        """
        return self.bb_center_

    @property
    def centroid(self):
        """:obj:`numpy.ndarray` of float :
        The 3D location of the mesh's vertex mean (read-only).
        """
        return self.centroid_

    ##################################################################
    # Public Class Methods
    ##################################################################

    def min_coords(self):
        """Returns the minimum coordinates of the mesh.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the minimal
            x, y, and z coordinates represented in the mesh.
        """
        return np.min(self.vertices_, axis=0)

    def max_coords(self):
        """Returns the maximum coordinates of the mesh.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the minimal
            x, y, and z coordinates represented in the mesh.
        """
        return np.max(self.vertices_, axis=0)

    def bounding_box(self):
        """Returns the mesh's bounding box corners.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            A 2-tuple of 3-ndarrays of floats. The first 3-array
            contains the vertex of the smallest corner of the bounding box,
            and the second 3-array contains the largest corner of the bounding
            box.
        """
        return self.min_coords(), self.max_coords()

    def bounding_box_mesh(self):
        """Returns the mesh bounding box as a mesh.

        Returns
        -------
        :obj:`Mesh3D`
            A Mesh3D representation of the mesh's bounding box.
        """
        min_vert, max_vert = self.bounding_box()
        xs, ys, zs = list(zip(max_vert, min_vert))
        vertices = []
        for x in xs:
            for y in ys:
                for z in zs:
                    vertices.append([x, y, z])
        triangles = (np.array([
            [5, 7, 3], [5, 3, 1],
            [2, 4, 8], [2, 8, 6],
            [6, 8, 7], [6, 7, 5],
            [1, 3, 4], [1, 4, 2],
            [6, 5, 1], [6, 1, 2],
            [7, 8, 4], [7, 4, 3],
        ]) - 1)
        return Mesh3D(vertices, triangles)

    def principal_dims(self):
        """Returns the maximal span of the mesh's coordinates.

        The maximal span is the maximum coordinate value minus
        the minimal coordinate value in each principal axis.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the maximal
            x, y, and z spans of the mesh.
        """
        return self.max_coords() - self.min_coords()

    def support(self, direction):
        """Returns the support function in the given direction

        Parameters
        ----------
        direction : :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that is a unit vector in
            the direction of the desired support.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the support.
        """
        ip = self.vertices_.dot(direction)
        index = np.where(ip == np.max(ip))[0][0]
        x0 = self.vertices_[index, :]
        n = direction
        com_proj = x0.dot(n) * n
        return com_proj

    def tri_centers(self):
        """Returns an array of the triangle centers as 3D points.

        Returns
        -------
        :obj:`numpy.ndarray` of :obj:`numpy.ndarray` of float
            An ndarray of 3-ndarrays of floats, where each 3-ndarray
            represents the 3D point at the center of the corresponding
            mesh triangle.
        """
        centers = []
        for tri in self.triangles_:
            centers.append(self._center_of_tri(tri))
        return np.array(centers)

    def tri_normals(self, align_to_hull=False):
        """Returns a list of the triangle normals.

        Parameters
        ----------
        align_to_hull : bool
            If true, we re-orient the normals to point outward from
            the mesh by using the convex hull.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A #triangles by 3 array of floats, where each 3-ndarray
            represents the 3D normal vector of the corresponding triangle.
        """
        # compute normals
        v0 = self.vertices_[self.triangles_[:, 0], :]
        v1 = self.vertices_[self.triangles_[:, 1], :]
        v2 = self.vertices_[self.triangles_[:, 2], :]
        n = np.cross(v1 - v0, v2 - v0)
        normals = n / np.tile(np.linalg.norm(n, axis=1)[:, np.newaxis], [1, 3])

        # reverse normal based on alignment with convex hull
        if align_to_hull:
            tri_centers = self.tri_centers()
            hull = ss.ConvexHull(tri_centers)
            hull_tris = hull.simplices
            hull_vertex_ind = hull_tris[0][0]
            hull_vertex = tri_centers[hull_vertex_ind]
            hull_vertex_normal = normals[hull_vertex_ind]
            v = hull_vertex.reshape([1, 3])
            n = hull_vertex_normal
            ip = (tri_centers - np.tile(hull_vertex,
                                        [tri_centers.shape[0], 1])).dot(n)
            if ip[0] > 0:
                normals = -normals
        return normals

    def surface_area(self):
        """Return the surface area of the mesh.

        Returns
        -------
        float
            The surface area of the mesh.
        """
        if self.surface_area_ is None:
            area = 0.0
            for tri in self.triangles:
                tri_area = self._area_of_tri(tri)
                area += tri_area
            self.surface_area_ = area
        return self.surface_area_

    def total_volume(self):
        """Return the total volume of the mesh.

        Returns
        -------
        float
            The total volume of the mesh.
        """
        total_volume = 0
        for tri in self.triangles_:
            volume = self._signed_volume_of_tri(tri)
            total_volume = total_volume + volume

        # Correct for flipped triangles
        if total_volume < 0:
            total_volume = -total_volume
        return total_volume

    def covariance(self):
        """Return the total covariance of the mesh's triangles.

        Returns
        -------
        float
            The total covariance of the mesh's triangles.
        """
        C_sum = np.zeros([3, 3])
        for tri in self.triangles_:
            C = self._covariance_of_tri(tri)
            C_sum = C_sum + C
        return C_sum

    def remove_bad_tris(self):
        """Remove triangles with out-of-bounds vertices from the mesh.
        """
        new_tris = []
        num_v = self.vertices_.shape[0]
        for t in self.triangles_:
            if (t[0] >= 0 and t[0] < num_v and
                    t[1] >= 0 and t[1] < num_v and
                    t[2] >= 0 and t[2] < num_v):
                new_tris.append(t)
        self.triangles = np.array(new_tris)

    def remove_unreferenced_vertices(self):
        """Remove any vertices that are not part of a triangular face.

        Note
        ----
        This method will fail if any bad triangles are present, so run
        remove_bad_tris() first if you're unsure if bad triangles are present.

        Returns
        -------
        bool
            Returns True if vertices were removed, False otherwise.

        """
        num_v = self.vertices_.shape[0]

        # Fill in a 1 for each referenced vertex
        reffed_array = np.zeros([num_v, 1])
        for f in self.triangles_:
            reffed_array[f[0]] = 1
            reffed_array[f[1]] = 1
            reffed_array[f[2]] = 1

        # Trim out vertices that are not referenced
        reffed_v_old_ind = np.where(reffed_array == 1)
        reffed_v_old_ind = reffed_v_old_ind[0]

        # Count number of referenced vertices before each index
        reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1

        try:
            self.vertices = self.vertices_[reffed_v_old_ind, :]
            if self.normals is not None:
                self.normals = self.normals[reffed_v_old_ind, :]
        except IndexError:
            return False

        # create new face indices
        new_triangles = []
        for f in self.triangles_:
            new_triangles.append([reffed_v_new_ind[f[0]],
                                  reffed_v_new_ind[f[1]],
                                  reffed_v_new_ind[f[2]]])
        self.triangles = np.array(new_triangles)
        return True

    def center_vertices_avg(self):
        """Center the mesh's vertices at the centroid.

        This shifts the mesh without rotating it so that
        the centroid (mean) of all vertices is at the origin.
        """
        centroid = np.mean(self.vertices_, axis=0)
        self.vertices = self.vertices_ - centroid

    def center_vertices_bb(self):
        """Center the mesh's vertices at the center of its bounding box.

        This shifts the mesh without rotating it so that
        the center of its bounding box is at the origin.
        """
        min_vertex = self.min_coords()
        max_vertex = self.max_coords()
        center = (max_vertex + min_vertex) / 2
        self.vertices = self.vertices_ - center

    def center_vertices(self):
        """Center the mesh's vertices on the mesh center of mass.

        This shifts the mesh without rotating it so that
        the center of its bounding box is at the origin.
        """
        self.vertices = self.vertices_ - self.center_of_mass_
        self.trimesh_ = None  # flag re-comp of trimesh

    def normalize_vertices(self):
        """Normalize the mesh's orientation along its principal axes.

        Transforms the vertices and normals of the mesh
        such that the origin of the resulting mesh's coordinate frame
        is at the center of the bounding box and the principal axes (as determined
        from PCA) are aligned with the vertical Z, Y, and X axes in that order.
        """

        self.center_vertices_bb()

        # Find principal axes
        pca = sklearn.decomposition.PCA(n_components=3)
        pca.fit(self.vertices_)

        # Count num vertices on side of origin wrt principal axes
        # to determine correct orientation
        comp_array = pca.components_
        norm_proj = self.vertices_.dot(comp_array.T)
        opposite_aligned = np.sum(norm_proj < 0, axis=0)
        same_aligned = np.sum(norm_proj >= 0, axis=0)

        # create rotation from principal axes to standard basis
        z_axis = comp_array[0, :]
        y_axis = comp_array[1, :]
        if opposite_aligned[2] > same_aligned[2]:
            z_axis = -z_axis
        if opposite_aligned[1] > same_aligned[1]:
            y_axis = -y_axis
        x_axis = np.cross(y_axis, z_axis)
        R_pc_obj = np.c_[x_axis, y_axis, z_axis]

        # rotate vertices, normals and reassign to the mesh
        self.vertices = (R_pc_obj.T.dot(self.vertices.T)).T
        self.center_vertices_bb()

        # TODO JEFF LOOK HERE (BUG IN INITIAL CODE FROM MESHPROCESSOR)
        if self.normals_ is not None:
            self.normals = (R_pc_obj.T.dot(self.normals.T)).T

    def compute_vertex_normals(self):
        """ Get normals from triangles"""
        normals = []
        # weighted average of triangle normal for each vertex
        for i in range(len(self.vertices)):
            inds = np.where(self.triangles == i)
            tris = self.triangles[inds[0], :]
            normal = np.zeros(3)
            for tri in tris:
                # compute triangle normal
                t = self.vertices[tri, :]
                v0 = t[1, :] - t[0, :]
                v1 = t[2, :] - t[0, :]
                if np.linalg.norm(v0) == 0:
                    continue
                v0 = v0 / np.linalg.norm(v0)
                if np.linalg.norm(v1) == 0:
                    continue
                v1 = v1 / np.linalg.norm(v1)
                n = np.cross(v0, v1)
                if np.linalg.norm(n) == 0:
                    continue
                n = n / np.linalg.norm(n)

                # compute weight by area of triangle
                w_area = self._area_of_tri(tri)

                # compute weight by edge angle
                vertex_ind = np.where(tri == i)[0][0]
                if vertex_ind == 0:
                    e0 = t[1, :] - t[0, :]
                    e1 = t[2, :] - t[0, :]
                elif vertex_ind == 1:
                    e0 = t[0, :] - t[1, :]
                    e1 = t[2, :] - t[1, :]
                elif vertex_ind == 2:
                    e0 = t[0, :] - t[2, :]
                    e1 = t[1, :] - t[2, :]
                if np.linalg.norm(e0) == 0:
                    continue
                if np.linalg.norm(e1) == 0:
                    continue
                e0 = e0 / np.linalg.norm(e0)
                e1 = e1 / np.linalg.norm(e1)
                w_angle = np.arccos(e0.dot(e1))

                # weighted update
                # www.bytehazard.com/articles/vertnorm.html
                normal += w_area * w_angle * n

            # normalize
            if np.linalg.norm(normal) == 0:
                normal = np.array([1, 0, 0])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)

        # set numpy array
        self.normals = np.array(normals)

        # reverse normals based on alignment with convex hull
        hull = ss.ConvexHull(self.vertices)
        hull_tris = hull.simplices.tolist()
        hull_vertex_inds = np.unique(hull_tris)

        num_aligned = 0
        num_misaligned = 0
        for hull_vertex_ind in hull_vertex_inds:
            hull_vertex = self.vertices[hull_vertex_ind, :]
            hull_vertex_normal = normals[hull_vertex_ind]
            ip = (hull_vertex - self.vertices).dot(hull_vertex_normal)
            num_aligned += np.sum(ip > 0)
            num_misaligned += np.sum(ip <= 0)

        if num_misaligned > num_aligned:
            self.normals = -self.normals

    def flip_normals(self):
        """ Flips the mesh normals. """
        if self.normals is not None:
            self.normals = -self.normals
            return True
        return False

    def scale_principal_eigenvalues(self, new_evals):
        self.normalize_vertices()

        pca = sklearn.decomposition.PCA(n_components=3)
        pca.fit(self.vertices_)

        evals = pca.explained_variance_
        if len(new_evals) == 3:
            self.vertices[:, 0] *= new_evals[2] / np.sqrt(evals[2])
            self.vertices[:, 1] *= new_evals[1] / np.sqrt(evals[1])
            self.vertices[:, 2] *= new_evals[0] / np.sqrt(evals[0])
        elif len(new_evals) == 2:
            self.vertices[:, 1] *= new_evals[1] / np.sqrt(evals[1])
            self.vertices[:, 2] *= new_evals[0] / np.sqrt(evals[0])
        elif len(new_evals) == 1:
            self.vertices[:, 0] *= new_evals[0] / np.sqrt(evals[0])
            self.vertices[:, 1] *= new_evals[0] / np.sqrt(evals[0])
            self.vertices[:, 2] *= new_evals[0] / np.sqrt(evals[0])

        self.center_vertices_bb()
        return evals

    def copy(self):
        """Return a copy of the mesh.

        Note
        ----
        This method only copies the vertices and triangles of the mesh.
        """
        return Mesh3D(np.copy(self.vertices_), np.copy(self.triangles_))

    def subdivide(self, min_tri_length=np.inf):
        """Return a copy of the mesh that has been subdivided by one iteration.

        Note
        ----
        This method only copies the vertices and triangles of the mesh.
        """
        new_vertices = self.vertices.tolist()
        old_triangles = self.triangles.tolist()

        new_triangles = []
        tri_queue = queue.Queue()

        for j, triangle in enumerate(old_triangles):
            tri_queue.put((j, triangle))

        num_subdivisions_per_tri = np.zeros(len(old_triangles))
        while not tri_queue.empty():
            tri_index_pair = tri_queue.get()
            j = tri_index_pair[0]
            triangle = tri_index_pair[1]

            if (np.isinf(min_tri_length) and num_subdivisions_per_tri[j] == 0) or \
                    (Mesh3D._max_edge_length(triangle, new_vertices) > min_tri_length):

                # subdivide
                t_vertices = np.array([new_vertices[i] for i in triangle])
                edge01 = 0.5 * (t_vertices[0, :] + t_vertices[1, :])
                edge12 = 0.5 * (t_vertices[1, :] + t_vertices[2, :])
                edge02 = 0.5 * (t_vertices[0, :] + t_vertices[2, :])

                i_01 = len(new_vertices)
                i_12 = len(new_vertices) + 1
                i_02 = len(new_vertices) + 2
                new_vertices.append(edge01)
                new_vertices.append(edge12)
                new_vertices.append(edge02)

                num_subdivisions_per_tri[j] += 1

                for triplet in [[triangle[0], i_01, i_02],
                                [triangle[1], i_12, i_01],
                                [triangle[2], i_02, i_12],
                                [i_01, i_12, i_02]]:
                    tri_queue.put((j, triplet))

            else:
                # add to final list
                new_triangles.append(triangle)

        return Mesh3D(np.array(new_vertices), np.array(new_triangles),
                      center_of_mass=self.center_of_mass)

    def transform(self, T):
        """Return a copy of the mesh that has been transformed by T.

        Parameters
        ----------
        T : :obj:`RigidTransform`
            The RigidTransform by which the mesh is transformed.

        Note
        ----
        This method only copies the vertices and triangles of the mesh.
        """
        vertex_cloud = PointCloud(self.vertices_.T, frame=T.from_frame)
        vertex_cloud_tf = T * vertex_cloud
        vertices = vertex_cloud_tf.data.T
        if self.normals_ is not None:
            normal_cloud = NormalCloud(self.normals_.T, frame=T.from_frame)
            normal_cloud_tf = T * normal_cloud
            normals = normal_cloud_tf.data.T
        com = Point(self.center_of_mass_, frame=T.from_frame)
        com_tf = T * com

        if self.normals_ is not None:
            return Mesh3D(vertices.copy(), self.triangles.copy(), normals=normals.copy(), center_of_mass=com_tf.data)
        return Mesh3D(vertices.copy(), self.triangles.copy(), center_of_mass=com_tf.data)

    def update_tf(self, delta_T):
        """ Updates the mesh transformation. """
        new_T_obj_world = self.T_obj_world * delta_T.inverse().as_frames('obj', 'obj')
        return Mesh3D(self.vertices, self.triangles, normals=self.normals, trimesh=self.trimesh,
                      T_obj_world=new_T_obj_world)

    def random_points(self, n_points):
        """Generate uniformly random points on the surface of the mesh.

        Parameters
        ----------
        n_points : int
            The number of random points to generate.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A n_points by 3 ndarray that contains the sampled 3D points.
        """
        probs = self._tri_area_percentages()
        tri_inds = np.random.choice(list(range(len(probs))), n_points, p=probs)
        points = []
        for tri_ind in tri_inds:
            tri = self.triangles[tri_ind]
            points.append(self._rand_point_on_tri(tri))
        return np.array(points)

    def ray_intersections(self, ray, point, distance):
        """Returns a list containing the indices of the triangles that
        are intersected by the given ray emanating from the given point
        within some distance.
        """
        ray = ray / np.linalg.norm(ray)
        norms = self.tri_normals()
        tri_point_pairs = []
        for i, tri in enumerate(self.triangles):
            if np.dot(ray, norms[i]) == 0.0:
                continue
            t = -1 * np.dot((point - self.vertices[tri[0]]), norms[i]) / (np.dot(ray, norms[i]))
            if (t > 0 and t <= distance):
                contact_point = point + t * ray
                tri_verts = [self.vertices[j] for j in tri]
                if Mesh3D._point_in_tri(tri_verts, contact_point):
                    tri_point_pairs.append((i, contact_point))
        return tri_point_pairs

    def get_T_surface_obj(self, T_obj_surface, delta=0.0):
        """ Gets the transformation that puts the object resting exactly on
        the z=delta plane

        Parameters
        ----------
        T_obj_surface : :obj:`RigidTransform`
            The RigidTransform by which the mesh is transformed.
        delta : float
            Z-coordinate to rest the mesh on

        Note
        ----
        This method copies the vertices and triangles of the mesh.
        """
        T_obj_surface_ori = T_obj_surface.copy()
        T_obj_surface_ori.translation = np.zeros(3)
        obj_tf = self.transform(T_obj_surface_ori)
        mn, mx = obj_tf.bounding_box()

        z = mn[2]
        x0 = np.array([0, 0, -z + delta])

        T_obj_surface = RigidTransform(rotation=T_obj_surface_ori.rotation,
                                       translation=x0, from_frame='obj',
                                       to_frame='surface')
        return T_obj_surface

    def rescale_dimension(self, scale, scaling_type=ScalingTypeMin):
        """Rescales the vertex coordinates to scale using the given scaling_type.

        Parameters
        ----------
        scale : float
            The desired scaling factor of the selected dimension, if scaling_type
            is ScalingTypeMin, ScalingTypeMed, ScalingTypeMax, or
            ScalingTypeDiag. Otherwise, the overall scaling factor.

        scaling_type : int
            One of ScalingTypeMin, ScalingTypeMed, ScalingTypeMax,
            ScalingTypeRelative, or ScalingTypeDiag.
            ScalingTypeMin scales the smallest vertex extent (X, Y, or Z)
            by scale, ScalingTypeMed scales the median vertex extent, and
            ScalingTypeMax scales the maximum vertex extent. ScalingTypeDiag
            scales the bounding box diagonal (divided by three), and
            ScalingTypeRelative provides absolute scaling.
        """
        vertex_extent = self.principal_dims()

        # Find minimal dimension
        relative_scale = 1.0
        if scaling_type == Mesh3D.ScalingTypeMin:
            dim = np.where(vertex_extent == np.min(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif scaling_type == Mesh3D.ScalingTypeMed:
            dim = np.where(vertex_extent == np.med(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif scaling_type == Mesh3D.ScalingTypeMax:
            dim = np.where(vertex_extent == np.max(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif scaling_type == Mesh3D.ScalingTypeRelative:
            relative_scale = 1.0
        elif scaling_type == Mesh3D.ScalingTypeDiag:
            diag = np.linalg.norm(vertex_extent)
            relative_scale = diag / 3.0  # make the gripper size exactly one third of the diagonal

        # Compute scale factor and rescale vertices
        scale_factor = scale / relative_scale
        self.vertices = scale_factor * self.vertices

    def rescale(self, scale_factor):
        """Rescales the vertex coordinates by scale_factor.

        Parameters
        ----------
        scale_factor : float
            The desired scale factor for the mesh's vertices.
        """
        self.vertices = scale_factor * self.vertices

    def convex_hull(self):
        """Return a 3D mesh that represents the convex hull of the mesh.
        """
        hull = ss.ConvexHull(self.vertices_)
        hull_tris = hull.simplices
        if self.normals_ is None:
            cvh_mesh = Mesh3D(self.vertices_.copy(), hull_tris.copy(), center_of_mass=self.center_of_mass_)
        else:
            cvh_mesh = Mesh3D(self.vertices_.copy(), hull_tris.copy(), normals=self.normals_.copy(),
                              center_of_mass=self.center_of_mass_)
        cvh_mesh.remove_unreferenced_vertices()
        return cvh_mesh

    def stable_poses(self, min_prob=0.0):
        """Computes all valid StablePose objects for the mesh.

        Parameters
        ----------
        min_prob : float
            stable poses that are less likely than this threshold will be discarded

        Returns
        -------
        :obj:`list` of :obj:`StablePose`
            A list of StablePose objects for the mesh.
        """
        # compute face dag if necessary
        if self.face_dag_ is None:
            self._compute_face_dag()
        cvh_mesh = self.face_dag_.mesh
        cvh_verts = self.face_dag_.mesh.vertices

        # propagate probabilities
        cm = self.center_of_mass
        prob_map = Mesh3D._compute_prob_map(list(self.face_dag_.nodes.values()), cvh_verts, cm)

        # compute stable poses
        stable_poses = []
        for face, p in list(prob_map.items()):
            x0 = cvh_verts[face[0]]
            r = cvh_mesh._compute_basis([cvh_verts[i] for i in face])
            if p > min_prob:
                stable_poses.append(sp.StablePose(p, r, x0, face=face))

        return stable_poses

    def resting_pose(self, T_obj_world, eps=1e-10):
        """ Returns the stable pose that the mesh will rest on if it lands
        on an infinite planar worksurface quasi-statically in the given
        transformation (only the rotation is used).

        Parameters
        ----------
        T_obj_world : :obj:`autolab_core.RigidTransform`
            transformation from object to table basis (z-axis upward) specifying the orientation of the mesh
        eps : float
            numeric tolerance in cone projection solver
        
        Returns
        -------
        :obj:`StablePose`
            stable pose specifying the face that the mesh will land on
        """
        # compute face dag if necessary
        if self.face_dag_ is None:
            self._compute_face_dag()

        # adjust transform to place mesh in contact with table
        T_obj_table = self.get_T_surface_obj(T_obj_world, delta=0.0)

        # transform mesh
        cvh_mesh = self.face_dag_.mesh
        cvh_verts = cvh_mesh.vertices
        mesh_tf = cvh_mesh.transform(T_obj_table)
        vertices_tf = mesh_tf.vertices

        # find the vertex with the minimum z value
        min_z = np.min(vertices_tf[:, 2])
        contact_ind = np.where(vertices_tf[:, 2] == min_z)[0]
        if contact_ind.shape[0] == 0:
            raise ValueError('Unable to find the vertex contacting the table!')
        vertex_ind = contact_ind[0]

        # project the center of mass onto the table plane
        table_tri = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0]])
        proj_cm = Mesh3D._proj_point_to_plane(table_tri, self.center_of_mass)
        contact_vertex = vertices_tf[vertex_ind]
        v_cm = proj_cm - contact_vertex
        v_cm = v_cm[:2]

        # compute which face the vertex will topple onto
        # break loop when topple tri is found        
        topple_tri = None
        neighboring_tris = self.face_dag_.vertex_to_tri[vertex_ind]
        random.shuffle(neighboring_tris)
        for neighboring_tri in neighboring_tris:
            # find indices of other two vertices
            ind = [0, 1, 2]
            for i, v in enumerate(neighboring_tri):
                if np.allclose(contact_vertex, vertices_tf[v]):
                    ind.remove(i)

            # form edges in table plane
            i1 = neighboring_tri[ind[0]]
            i2 = neighboring_tri[ind[1]]
            v1 = Mesh3D._proj_point_to_plane(table_tri, vertices_tf[i1])
            v2 = Mesh3D._proj_point_to_plane(table_tri, vertices_tf[i2])
            u1 = v1 - contact_vertex
            u2 = v2 - contact_vertex
            U = np.array([u1[:2], u2[:2]]).T

            # solve linear subproblem to find cone coefficients
            try:
                alpha = np.linalg.solve(U + eps * np.eye(2), v_cm)

                # exit loop with topple tri if found
                if np.all(alpha >= 0):
                    tri_normal = cvh_mesh._compute_basis([cvh_verts[i] for i in neighboring_tri])[2, :]
                    if tri_normal[2] < 0:
                        tri_normal = -tri_normal

                    # check whether lower
                    lower = True
                    tri_center = np.mean([vertices_tf[i] for i in neighboring_tri], axis=0)
                    if topple_tri is not None:
                        topple_tri_center = np.mean([vertices_tf[i] for i in topple_tri], axis=0)
                        lower = (tri_normal.dot(topple_tri_center - tri_center) > 0)
                    if lower:
                        topple_tri = neighboring_tri

            except np.linalg.LinAlgError:
                logging.warning('Failed to solve linear system')

        # check solution
        if topple_tri is None:
            raise ValueError('Failed to find a valid topple triangle')

        # compute the face that the mesh will eventually rest on
        # by following the child nodes to a sink
        cur_node = self.face_dag_.nodes[tuple(topple_tri)]
        visited = []
        while not cur_node.is_sink:
            if cur_node in visited:
                raise ValueError('Found loop!')
            visited.append(cur_node)
            cur_node = cur_node.children[0]

        # create stable pose
        resting_face = cur_node.face
        x0 = cvh_verts[vertex_ind]
        R = cvh_mesh._compute_basis([cvh_verts[i] for i in resting_face])

        # align with axes with the original pose
        best_theta = 0
        best_dot = 0
        cur_theta = 0
        delta_theta = 0.01
        px = R[:, 0].copy()
        px[2] = 0
        py = R[:, 1].copy()
        py[2] = 0
        align_x = True
        if np.linalg.norm(py) > np.linalg.norm(px):
            align_x = False
        while cur_theta <= 2 * np.pi:
            Rz = RigidTransform.z_axis_rotation(cur_theta)
            Rp = Rz.dot(R)
            dot_prod = Rp[:, 0].dot(T_obj_world.x_axis)
            if not align_x:
                dot_prod = Rp[:, 1].dot(T_obj_world.y_axis)
            if dot_prod > best_dot:
                best_dot = dot_prod
                best_theta = cur_theta
            cur_theta += delta_theta
        R = RigidTransform.z_axis_rotation(best_theta).dot(R)
        return sp.StablePose(0.0, R, x0, face=resting_face)

    def merge(self, other_mesh):
        """ Combines this mesh with another mesh.
        
        Parameters
        ----------
        other_mesh : :obj:`Mesh3D`
            the mesh to combine with

        Returns
        -------
        :obj:`Mesh3D`
            merged mesh
        """
        total_vertices = self.num_vertices + other_mesh.num_vertices
        total_triangles = self.num_triangles + other_mesh.num_triangles
        combined_vertices = np.zeros([total_vertices, 3])
        combined_triangles = np.zeros([total_triangles, 3])

        combined_vertices[:self.num_vertices, :] = self.vertices
        combined_vertices[self.num_vertices:, :] = other_mesh.vertices

        combined_triangles[:self.num_triangles, :] = self.triangles
        combined_triangles[self.num_triangles:, :] = other_mesh.triangles + self.num_vertices

        combined_normals = None
        if self.normals is not None and other_mesh.normals is not None:
            combined_normals = np.zeros([total_vertices, 3])
            combined_normals[:self.num_vertices, :] = self.normals
            combined_normals[self.num_vertices:, :] = other_mesh.normals
        return Mesh3D(combined_vertices, combined_triangles.astype(np.int32), combined_normals)

    def flip_tri_orientation(self):
        """ Flips the orientation of all triangles. """
        new_tris = self.triangles
        new_tris[:, 1] = self.triangles[:, 2]
        new_tris[:, 2] = self.triangles[:, 1]
        return Mesh3D(self.vertices, new_tris, self.normals,
                      center_of_mass=self.center_of_mass)

    def find_contact(self, origin, direction):
        """ Finds the contact location with the mesh, if it exists. """
        # create points
        origin_world = Point(origin, frame='world')
        direction_world = Direction(direction, frame='world')

        # find contact using trimesh ray intersector
        origin_obj = self.T_obj_world.inverse() * origin_world
        direction_obj = self.T_obj_world.inverse() * direction_world
        locations, _, tri_indices = self.trimesh.ray.intersects_location([origin_obj.data], [direction_obj.data])

        if len(locations) == 0:
            return None, None

        # return closest point
        dists = np.linalg.norm(locations - origin_obj.data, axis=1)
        closest_ind = np.where(dists == np.min(dists))[0][0]
        point_obj = Point(locations[closest_ind, :], frame='obj')
        normal_obj = Direction(self.trimesh.face_normals[tri_indices[closest_ind], :], frame='obj')
        point_world = self.T_obj_world * point_obj
        normal_world = self.T_obj_world * normal_obj

        return point_world.data, normal_world.data

    def visualize(self, color=(0.5, 0.5, 0.5), style='surface', opacity=1.0):
        """Plots visualization of mesh using MayaVI.

        Parameters
        ----------
        color : :obj:`tuple` of float
            3-tuple of floats in [0,1] to give the mesh's color

        style : :obj:`str`
            Either 'surface', which produces an opaque surface, or
            'wireframe', which produces a wireframe.

        opacity : float
            A value in [0,1] indicating the opacity of the mesh.
            Zero is transparent, one is opaque.

        Returns
        -------
        :obj:`mayavi.modules.surface.Surface`
            The displayed surface.
        """
        surface = mv.triangular_mesh(self.vertices_[:, 0],
                                     self.vertices_[:, 1],
                                     self.vertices_[:, 2],
                                     self.triangles_, representation=style,
                                     color=color, opacity=opacity)
        return surface

    @staticmethod
    def load(filename, cache_dir, preproc_script=None):
        """Load a mesh from a file.

        Note
        ----
        If the mesh is not already in .obj format, this requires
        the installation of meshlab. Meshlab has a command called
        meshlabserver that is used to convert the file into a .obj format.

        Parameters
        ----------
        filename : :obj:`str`
            Path to mesh file.
        cache_dir : :obj:`str`
            A directory to store a converted .obj file in, if
            the file isn't already in .obj format.
        preproc_script : :obj:`str`
            The path to an optional script to run before converting
            the mesh file to .obj if necessary.

        Returns
        -------
        :obj:`Mesh3D`
            A 3D mesh object read from the file.
        """
        file_path, file_root = os.path.split(filename)
        file_root, file_ext = os.path.splitext(file_root)
        obj_filename = filename

        if file_ext != Mesh3D.OBJ_EXT:
            obj_filename = os.path.join(cache_dir, file_root + Mesh3D.PROC_TAG + Mesh3D.OBJ_EXT)
            if preproc_script is None:
                meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' % (filename, obj_filename)
            else:
                meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\" -s \"%s\"' % (
                filename, obj_filename, preproc_script)
            os.system(meshlabserver_cmd)

        if not os.path.exists(obj_filename):
            raise ValueError('Unable to open file %s. It may not exist or meshlab may not be installed.' % (filename))

        # Read mesh from obj file
        return obj_file.ObjFile(obj_filename).read()

    @property
    def trimesh(self):
        """ Convert to trimesh. """
        if self.trimesh_ is None:
            self.trimesh_ = tm.Trimesh(vertices=self.vertices,
                                       faces=self.triangles,
                                       vertex_normals=self.normals)
        return self.trimesh_

    @property
    def is_watertight(self):
        return self.trimesh.is_watertight

    @property
    def T_obj_world(self):
        """ Return pose. """
        return self.T_obj_world_

    ##################################################################
    # Private Class Methods
    ##################################################################

    def _compute_mass(self):
        """Computes the mesh mass.

        Note
        ----
            Only works for watertight meshes.

        Returns
        -------
        float
            The mass of the mesh.
        """
        return self.density_ * self.total_volume()

    def _compute_inertia(self):
        """Computes the mesh inertia matrix.

        Note
        ----
            Only works for watertight meshes.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3x3 inertial matrix.
        """
        C = self.covariance()
        return self.density_ * (np.trace(C) * np.eye(3) - C)

    def _compute_bb_center(self):
        """Computes the center point of the bounding box.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            3-ndarray of floats that contains the coordinates
            of the center of the bounding box.
        """

        bb_center = (self.min_coords() + self.max_coords()) / 2.0
        return bb_center

    def _compute_com_uniform(self):
        """Computes the center of mass using a uniform mass distribution assumption.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            3-ndarray of floats that contains the coordinates
            of the center of mass.
        """
        """
        total_volume = 0
        weighted_point_sum = np.zeros([1, 3])
        for tri in self.triangles_:
            volume = self._signed_volume_of_tri(tri)
            center = self._center_of_tri(tri)
            weighted_point_sum = weighted_point_sum + volume * center
            total_volume = total_volume + volume
        center_of_mass = weighted_point_sum / total_volume
        return center_of_mass[0]
        """
        return self.trimesh.center_mass

    def _compute_centroid(self):
        """Computes the centroid (mean) of the mesh's vertices.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            3-ndarray of floats that contains the coordinates
            of the centroid.
        """
        return np.mean(self.vertices_, axis=0)

    def _signed_volume_of_tri(self, tri):
        """Return the signed volume of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute a signed volume.

        Returns
        -------
        float
            The signed volume associated with the triangle.
        """
        v1 = self.vertices_[tri[0], :]
        v2 = self.vertices_[tri[1], :]
        v3 = self.vertices_[tri[2], :]

        volume = (1.0 / 6.0) * (v1.dot(np.cross(v2, v3)))
        return volume

    def _center_of_tri(self, tri):
        """Return the center of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute a signed volume.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3D point at the center of the triangle
        """
        v1 = self.vertices_[tri[0], :]
        v2 = self.vertices_[tri[1], :]
        v3 = self.vertices_[tri[2], :]
        center = (1.0 / 3.0) * (v1 + v2 + v3)
        return center

    def _covariance_of_tri(self, tri):
        """Return the covariance matrix of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute a covariance.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3x3 covariance matrix of the given triangle.
        """
        v1 = self.vertices_[tri[0], :]
        v2 = self.vertices_[tri[1], :]
        v3 = self.vertices_[tri[2], :]

        A = np.zeros([3, 3])
        A[:, 0] = v1 - self.center_of_mass_
        A[:, 1] = v2 - self.center_of_mass_
        A[:, 2] = v3 - self.center_of_mass_
        C = np.linalg.det(A) * A.dot(Mesh3D.C_canonical).dot(A.T)
        return C

    def _area_of_tri(self, tri):
        """Return the area of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute an area.

        Returns
        -------
        float
            The area of the triangle.
        """
        verts = [self.vertices[i] for i in tri]
        ab = verts[1] - verts[0]
        ac = verts[2] - verts[0]
        return 0.5 * np.linalg.norm(np.cross(ab, ac))

    def _tri_area_percentages(self):
        """Return a list of the percent area each triangle contributes to the
        mesh's surface area.

        Returns
        -------
        :obj:`list` of float
            A list of percentages in [0,1] for each face that represents its
            total contribution to the area of the mesh.
        """
        probs = []
        area = 0.0
        for tri in self.triangles:
            tri_area = self._area_of_tri(tri)
            probs.append(tri_area)
            area += tri_area
        probs = probs / area
        return probs

    def _rand_point_on_tri(self, tri):
        """Return a random point on the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute an area.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3D point on the triangle.
        """
        verts = [self.vertices[i] for i in tri]
        r1 = np.sqrt(np.random.uniform())
        r2 = np.random.uniform()
        p = (1 - r1) * verts[0] + r1 * (1 - r2) * verts[1] + r1 * r2 * verts[2]
        return p

    def _compute_proj_area(self, verts):
        """Projects vertices onto the unit sphere from the center of mass
        and computes the projected area.

        Parameters
        ----------
        verts : `list` of `numpy.ndarray` of float
            List of 3-ndarrays of floats that represent the vertices to be
            projected onto the unit sphere.

        Returns
        -------
        float
            The total projected area on the unit sphere.
        """
        cm = self.center_of_mass
        angles = []

        proj_verts = [(v - cm) / np.linalg.norm(v - cm) for v in verts]

        a = math.acos(min(1, max(-1, np.dot(proj_verts[0], proj_verts[1]) /
                                 (np.linalg.norm(proj_verts[0]) * np.linalg.norm(proj_verts[1])))))
        b = math.acos(min(1, max(-1, np.dot(proj_verts[0], proj_verts[2]) /
                                 (np.linalg.norm(proj_verts[0]) * np.linalg.norm(proj_verts[2])))))
        c = math.acos(min(1, max(-1, np.dot(proj_verts[1], proj_verts[2]) /
                                 (np.linalg.norm(proj_verts[1]) * np.linalg.norm(proj_verts[2])))))
        s = (a + b + c) / 2

        try:
            return 4 * math.atan(math.sqrt(math.tan(s / 2) * math.tan((s - a) / 2) *
                                           math.tan((s - b) / 2) * math.tan((s - c) / 2)))
        except:
            s = s + 0.001
            return 4 * math.atan(math.sqrt(math.tan(s / 2) * math.tan((s - a) / 2) *
                                           math.tan((s - b) / 2) * math.tan((s - c) / 2)))

    def _compute_basis(self, face_verts):
        """Computes axes for a transformed basis relative to the plane in which input vertices lie.

        Parameters
        ----------
        face_verts : :obj:`numpy.ndarray` of float
            A set of three 3D points that form a plane.

        Returns:
        :obj:`numpy.ndarray` of float
            A 3-by-3 ndarray whose rows are the new basis. This matrix
            can be applied to the mesh to rotate the mesh to lie flat
            on the input face.
        """
        centroid = np.mean(face_verts, axis=0)

        z_o = np.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
        z_o = z_o / np.linalg.norm(z_o)

        # Ensure that all vertices are on the positive halfspace (aka above the table)
        dot_product = (self.vertices - centroid).dot(z_o)
        dot_product[np.abs(dot_product) < 1e-10] = 0.0
        if np.any(dot_product < 0):
            z_o = -z_o

        x_o = np.array([-z_o[1], z_o[0], 0])
        if np.linalg.norm(x_o) == 0.0:
            x_o = np.array([1, 0, 0])
        else:
            x_o = x_o / np.linalg.norm(x_o)
        y_o = np.cross(z_o, x_o)
        y_o = y_o / np.linalg.norm(y_o)

        R = np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])

        # rotate the vertices and then align along the principal axes
        rotated_vertices = R.dot(self.vertices.T)
        xy_components = rotated_vertices[:2, :].T

        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(xy_components)
        comp_array = pca.components_
        x_o = R.T.dot(np.array([comp_array[0, 0], comp_array[0, 1], 0]))
        y_o = np.cross(z_o, x_o)
        return np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])

    def _compute_face_dag(self):
        """ Computes a directed acyclic graph (DAG) specifying the
        toppling structure of the mesh faces by:
            1) Computing the mesh convex hull
            2) Creating maps from vertices and edges to the triangles that share them 
            3) Connecting each triangle in the convex hull to the face it will topple to, if landed on
        Modifies the class variable self.face_dag_.
        """
        # compute convex hull
        cm = self.center_of_mass
        cvh_mesh = self.convex_hull()
        cvh_tris = cvh_mesh.triangles
        cvh_verts = cvh_mesh.vertices

        # create vertex and edge maps, and create nodes of graph
        nodes = {}  # mapping from triangle tuples to GraphVertex objects
        vertex_to_tri = {}  # mapping from vertex indidces to adjacent triangle lists
        edge_to_tri = {}  # mapping from edge tuples to adjacent triangle lists

        for tri in cvh_tris:
            # add vertex to tri mapping
            for v in tri:
                if v in vertex_to_tri:
                    vertex_to_tri[v] += [tuple(tri)]
                else:
                    vertex_to_tri[v] = [tuple(tri)]

            # add edges to tri mapping
            tri_verts = [cvh_verts[i] for i in tri]
            s1 = Mesh3D._Segment(tri_verts[0], tri_verts[1])
            s2 = Mesh3D._Segment(tri_verts[0], tri_verts[2])
            s3 = Mesh3D._Segment(tri_verts[1], tri_verts[2])
            for seg in [s1, s2, s3]:
                k = seg.tup
                if k in edge_to_tri:
                    edge_to_tri[k] += [tuple(tri)]
                else:
                    edge_to_tri[k] = [tuple(tri)]

            # add triangle to graph with prior probability estimate
            p = self._compute_proj_area(tri_verts) / (4 * math.pi)
            nodes[tuple(tri)] = Mesh3D._GraphVertex(p, tri)

        # connect nodes in the graph based on geometric toppling criteria
        # a directed edge between two graph nodes implies that landing on one face will lead to toppling onto its successor
        # an outdegree of 0 for any graph node implies it is a sink (the object will come to rest if it topples to this face)
        for tri in cvh_tris:
            # vertices
            tri_verts = [cvh_verts[i] for i in tri]

            # project the center of mass onto the triangle
            proj_cm = Mesh3D._proj_point_to_plane(tri_verts, cm)

            # update list of top vertices, add edges between vertices as needed
            if not Mesh3D._point_in_tri(tri_verts, proj_cm):
                # form segment objects
                s1 = Mesh3D._Segment(tri_verts[0], tri_verts[1])
                s2 = Mesh3D._Segment(tri_verts[0], tri_verts[2])
                s3 = Mesh3D._Segment(tri_verts[1], tri_verts[2])

                # compute the closest edges
                closest_edges = Mesh3D._closest_segment(proj_cm, [s1, s2, s3])

                # choose the closest edge based on the midpoint of the triangle segments
                if len(closest_edges) == 1:
                    closest_edge = closest_edges[0]
                else:
                    closest_edge = Mesh3D._closer_segment(proj_cm, closest_edges[0], closest_edges[1])

                    # compute the topple face from the closest edge
                for face in edge_to_tri[closest_edge.tup]:
                    if list(face) != list(tri):
                        topple_face = face
                predecessor = nodes[tuple(tri)]
                successor = nodes[tuple(topple_face)]
                predecessor.add_edge(successor)

        # save to class variable
        self.face_dag_ = Mesh3D._FaceDAG(cvh_mesh, nodes, vertex_to_tri, edge_to_tri)

    class _Segment:
        """Object representation of a finite line segment in 3D space.

        Attributes
        ----------
        p1 : :obj:`numpy.ndarray` of float
            The first endpoint of the line segment
        p2 : :obj:`numpy.ndarray` of float
            The second endpoint of the line segment
        tup : :obj:`tuple` of :obj:`tuple` of float
            A tuple representation of the segment, with the two
            endpoints arranged in lexicographical order.
        """

        def __init__(self, p1, p2):
            """Creates a Segment with given endpoints.

            Parameters
            ----------
            p1 : :obj:`numpy.ndarray` of float
                The first endpoint of the line segment
            p2 : :obj:`numpy.ndarray` of float
                The second endpoint of the line segment
            """
            self.p1 = p1
            self.p2 = p2
            self.tup = self._ordered_tuple()

        def dist_to_point(self, point):
            """Computes the distance from the segment to the given 3D point.

            Parameters
            ----------
            point : :obj:`numpy.ndarray` of float
                The 3D point to measure distance to.

            Returns
            -------
            float
                The euclidean distance between the segment and the point.
            """
            p1, p2 = self.p1, self.p2
            ap = point - p1
            ab = p2 - p1
            proj_point = p1 + (np.dot(ap, ab) / np.dot(ab, ab)) * ab
            if self._contains_proj_point(proj_point):
                return np.linalg.norm(point - proj_point)
            else:
                return min(np.linalg.norm(point - p1),
                           np.linalg.norm(point - p2))

        def _contains_proj_point(self, point):
            """Is the given 3D point (assumed to be on the line that contains
            the segment) actually within the segment?

            Parameters
            ----------
            point : :obj:`numpy.ndarray` of float
                The 3D point to check against.

            Returns
            -------
            bool
                True if the point was within the segment or False otherwise.
            """
            p1, p2 = self.p1, self.p2
            return (point[0] >= min(p1[0], p2[0]) and point[0] <= max(p1[0], p2[0]) and
                    point[1] >= min(p1[1], p2[1]) and point[1] <= max(p1[1], p2[1]) and
                    point[2] >= min(p1[2], p2[2]) and point[2] <= max(p1[2], p2[2]))

        def _ordered_tuple(self):
            """Returns an ordered tuple that represents the segment.

            The points within are ordered lexicographically.

            Returns
            -------

            tup : :obj:`tuple` of :obj:`tuple` of float
                A tuple representation of the segment, with the two
                endpoints arranged in lexicographical order.
            """
            if (self.p1.tolist() > self.p2.tolist()):
                return (tuple(self.p1), tuple(self.p2))
            else:
                return (tuple(self.p2), tuple(self.p1))

    class _FaceDAG:
        """ A directed acyclic graph specifying the topppling dependency structure
        for faces of a given mesh geometry with a specific center of mass.
        Useful for quasi-static stable pose analysis.

        Attributes
        ----------
        mesh : :obj:`Mesh3D`
            the 3D triangular mesh that the DAG refers to (usually the convex hull) 
        nodes : :obj:`dict` mapping 3-`tuple` of integers (triangles) to :obj:`Mesh3D._GraphVertex`
            the nodes in the DAG
        vertex_to_tri : :obj:`dict` mapping :obj:`int` (vertex indices) to 3-`tuple` of integers (triangles)
        edge_to_tri : :obj:`dict` mapping 2-`tuple` of integers (edges) to 3-`tuple` of integers (triangles)
        """

        def __init__(self, mesh, nodes, vertex_to_tri, edge_to_tri):
            self.mesh = mesh
            self.nodes = nodes
            self.vertex_to_tri = vertex_to_tri
            self.edge_to_tri = edge_to_tri

    class _GraphVertex:
        """A directed graph vertex that links a probability to a face.
        """

        def __init__(self, probability, face):
            """Create a graph vertex with given probability and face.

            Parameters
            ----------
            probability : float
                Probability associated with this vertex.
            face : :obj:`numpy.ndarray` of int
                A 3x3 array that represents the face
                associated with this vertex. Each row is a list
                of vertices in one face.
            """
            self.probability = probability
            self.children = []
            self.parents = []
            self.face = face
            self.has_parent = False
            self.num_parents = 0
            self.sink = None

        @property
        def is_sink(self):
            return len(self.children) == 0

        def add_edge(self, child):
            """Connects this vertex to the input child vertex.

            Parameters
            ----------
            child : :obj:`_GraphVertex`
                The child to link to.
            """
            self.children.append(child)
            child.parents.append(self)
            child.has_parent = True
            child.num_parents += 1

    @staticmethod
    def _max_edge_length(tri, vertices):
        """Compute the maximum edge length of a triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle of interest.

        vertices : :obj:`numpy.ndarray` of `numpy.ndarray` of float
            The set of vertices which the face references.

        Returns
        -------
        float
            The max edge length of the triangle.
        """
        v0 = np.array(vertices[tri[0]])
        v1 = np.array(vertices[tri[1]])
        v2 = np.array(vertices[tri[2]])
        max_edge_len = max(np.linalg.norm(v1 - v0),
                           max(np.linalg.norm(v1 - v0),
                               np.linalg.norm(v2 - v1)))
        return max_edge_len

    @staticmethod
    def _proj_point_to_plane(tri_verts, point):
        """Project the given point onto the plane containing the three points in
        tri_verts.

        Parameters
        ----------
        tri_verts : :obj:`numpy.ndarray` of float
            A list of three 3D points that defines a plane.
        point : :obj:`numpy.ndarray` of float
            The 3D point to project onto the plane.
        """

        # Compute a normal vector to the triangle
        v0 = tri_verts[2] - tri_verts[0]
        v1 = tri_verts[1] - tri_verts[0]
        n = np.cross(v0, v1)
        n = n / np.linalg.norm(n)

        # Compute distance from the point to the triangle's plane
        # by projecting a vector from the plane to the point onto
        # the normal vector
        dist = np.dot(n, point - tri_verts[0])

        # Project the point back along the normal vector
        return (point - dist * n)

    @staticmethod
    def _point_in_tri(tri_verts, point):
        """Is the given point contained in the given triangle?

        Parameters
        ----------
        tri_verts : :obj:`list` of :obj:`numpy.ndarray` of float
            A list of three 3D points that definie a triangle.

        point : :obj:`numpy.ndarray` of float
            A 3D point that should be coplanar with the triangle.

        Returns
        -------
        bool
            True if the point is in the triangle, False otherwise.
        """
        # Implementation provided by http://blackpawn.com/texts/pointinpoly/

        # Compute vectors
        v0 = tri_verts[2] - tri_verts[0]
        v1 = tri_verts[1] - tri_verts[0]
        v2 = point - tri_verts[0]

        # Compute Dot Products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute Barycentric Coords
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0.0 and v >= 0.0 and u + v <= 1.0)

    @staticmethod
    def _closest_segment(point, line_segments):
        """Returns the finite line segment(s) the least distance from the input point.

        Parameters
        ----------
        point : :obj:`numpy.ndarray` of float
            The 3D point to measure distance to.
        line_segments: :obj:`list` of :obj:`_Segments`
            The list of line segments.

        Returns
        -------
        :obj:`list` of :obj:`_Segments`
            The list of line segments that were closest to the input point.
        """
        min_dist = sys.maxsize
        min_segs = []
        distances = []
        segments = []
        common_endpoint = None

        for segment in line_segments:
            dist = segment.dist_to_point(point)
            distances.append(dist)
            segments.append(segment)
            if dist < min_dist:
                min_dist = dist

        for i in range(len(distances)):
            if min_dist + 0.000001 >= distances[i]:
                min_segs.append(segments[i])

        return min_segs

    @staticmethod
    def _closer_segment(point, s1, s2):
        """ Compute which segment is closer to a given point by seeing
        which side of the midline between the two segments the point falls on.

        Parameters
        ----------
        point : :obj:`numpy.ndarray`
            3d array containing point projected onto plane spanned by s1, s1
        s1 : :obj:`Mesh3D._Segment`
            first segment to check
        s2 : :obj:`Mesh3D._Segment`
            second segment to check

        Returns
        -------
        :obj:`Mesh3D._Segment`
            best segment to check        
        """
        # find the shared vertex and compute the midline between the segments
        if np.allclose(s1.p1, s2.p1):
            p = s1.p1
            l1 = s1.p2 - p
            l2 = s2.p2 - p
        elif np.allclose(s1.p2, s2.p1):
            p = s1.p2
            l1 = s1.p1 - p
            l2 = s2.p2 - p
        elif np.allclose(s1.p1, s2.p2):
            p = s1.p1
            l1 = s1.p2 - p
            l2 = s2.p1 - p
        else:
            p = s1.p2
            l1 = s1.p1 - p
            l2 = s2.p1 - p
        v = point - p
        midline = 0.5 * (l1 + l2)

        # compute projection onto the midline
        if np.linalg.norm(midline) == 0:
            raise ValueError('Illegal triangle')
        alpha = midline.dot(v) / midline.dot(midline)
        w = alpha * midline

        # compute residual (component of query point orthogonal to midline)
        x = v - w

        # figure out which line is on the same side of the midline
        # as the residual
        d1 = x.dot(l1)
        d2 = x.dot(l2)
        closer_segment = s2
        if d1 > d2:
            closer_segment = s1
        return closer_segment

    @staticmethod
    def _compute_prob_map(vertices, cvh_verts, cm):
        """Creates a map from faces to static stability probabilities.

        Parameters
        ----------
        vertices : :obj:`list` of :obj:`_GraphVertex`

        Returns
        -------
        :obj:`dictionary` of :obj:`tuple` of int to float
            Maps tuple representations of faces to probabilities.
        """
        # follow the child nodes of each vertex until a sink, then add in the resting probability
        prob_mapping = {}
        for vertex in vertices:
            c = vertex
            visited = []
            while not c.is_sink:
                if c in visited:
                    break
                visited.append(c)
                c = c.children[0]

            if tuple(c.face) not in list(prob_mapping.keys()):
                prob_mapping[tuple(c.face)] = 0.0
            prob_mapping[tuple(c.face)] += vertex.probability
            vertex.sink = c

        # set resting probabilities of faces to zero
        for vertex in vertices:
            if not vertex.is_sink:
                prob_mapping[tuple(vertex.face)] = 0

        return prob_mapping


if __name__ == '__main__':
    pass
