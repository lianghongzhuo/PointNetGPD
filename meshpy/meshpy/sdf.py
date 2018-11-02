"""
Definition of SDF Class
Author: Sahaana Suri, Jeff Mahler, and Matt Matl

**Currently assumes clean input**
"""
from abc import ABCMeta, abstractmethod
import logging
import numpy as np
from numbers import Number

import time

from autolab_core import RigidTransform, SimilarityTransform, PointCloud, Point, NormalCloud


from sys import version_info
if version_info[0] != 3:
    range = xrange


# class Sdf(metaclass=ABCMeta):  # work for python3
class Sdf():
    """ Abstract class for signed distance fields.
    """
    __metaclass__ = ABCMeta
    ##################################################################
    # General SDF Properties
    ##################################################################
    @property
    def dimensions(self):
        """SDF dimension information.

        Returns
        -------
        :obj:`numpy.ndarray` of int
            The ndarray that contains the dimensions of the sdf.
        """
        return self.dims_

    @property
    def origin(self):
        """The location of the origin in the SDF grid.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 2- or 3-ndarray that contains the location of
            the origin of the mesh grid in real space.
        """
        return self.origin_

    @property
    def resolution(self):
        """The grid resolution (how wide each grid cell is).

        Returns
        -------
        float
            The width of each grid cell.
        """
        return self.resolution_

    @property
    def center(self):
        """Center of grid.

        This basically transforms the world frame to grid center.

        Returns
        -------
        :obj:`numpy.ndarray`
        """
        return self.center_

    @property
    def gradients(self):
        """Gradients of the SDF.

        Returns
        -------
        :obj:`list` of :obj:`numpy.ndarray` of float
            A list of ndarrays of the same dimension as the SDF. The arrays
            are in axis order and specify the gradients for that axis
            at each point.
        """
        return self.gradients_

    @property
    def data(self):
        """The SDF data.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 2- or 3-dimensional ndarray that holds the grid of signed
            distances.
        """
        return self.data_

    ##################################################################
    # General SDF Abstract Methods
    ##################################################################
    @abstractmethod
    def transform(self, tf):
        """Returns a new SDF transformed by similarity tf.
        """
        pass

    @abstractmethod
    def transform_pt_obj_to_grid(self, x_world, direction=False):
        """Transforms points from world frame to grid frame
        """
        pass

    @abstractmethod
    def transform_pt_grid_to_obj(self, x_grid, direction=False):
        """Transforms points from grid frame to world frame
        """
        pass

    @abstractmethod
    def surface_points(self):
        """Returns the points on the surface.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of int, :obj:`numpy.ndarray` of float
            The points on the surface and the signed distances at those points.
        """
        pass

    @abstractmethod
    def __getitem__(self, coords):
        """Returns the signed distance at the given coordinates.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The signed distance at the given coords (interpolated).
        """
        pass

    ##################################################################
    # Universal SDF Methods
    ##################################################################
    def transform_to_world(self):
        """Returns an sdf object with center in the world frame of reference.
        """
        return self.transform(self.pose_, scale=self.scale_)

    def center_world(self):
        """Center of grid (basically transforms world frame to grid center)
        """
        return self.transform_pt_grid_to_obj(self.center_)

    def on_surface(self, coords):
        """Determines whether or not a point is on the object surface.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        :obj:`tuple` of bool, float
            Is the point on the object's surface, and what
            is the signed distance at that point?
        """
        sdf_val = self[coords]
        if np.abs(sdf_val) < self.surface_thresh_:
            return True, sdf_val
        return False, sdf_val

    def is_out_of_bounds(self, coords):
        """Returns True if coords is an out of bounds access.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        bool
            Are the coordinates in coords out of bounds?
        """
        return np.array(coords < 0).any() or np.array(coords >= self.dims_).any()

    def _compute_gradients(self):
        """Computes the gradients of the SDF.

        Returns
        -------
        :obj:`list` of :obj:`numpy.ndarray` of float
            A list of ndarrays of the same dimension as the SDF. The arrays
            are in axis order and specify the gradients for that axis
            at each point.
        """
        self.gradients_ = np.gradient(self.data_)


class Sdf3D(Sdf):
    # static indexing vars
    num_interpolants = 8
    min_coords_x = [0, 2, 3, 5]
    max_coords_x = [1, 4, 6, 7]
    min_coords_y = [0, 1, 3, 6]
    max_coords_y = [2, 4, 5, 7]
    min_coords_z = [0, 1, 2, 4]
    max_coords_z = [3, 5, 6, 7]

    def __init__(self, sdf_data, origin, resolution, use_abs=False,
                 T_sdf_world=RigidTransform(from_frame='sdf', to_frame='world')):
        self.data_ = sdf_data
        self.origin_ = origin
        self.resolution_ = resolution
        self.dims_ = self.data_.shape

        # set up surface params
        self.surface_thresh_ = self.resolution_ * np.sqrt(2) / 2
        self.surface_points_ = None
        self.surface_points_w_ = None
        self.surface_vals_ = None
        self._compute_surface_points()

        # resolution is max dist from surface when surf is orthogonal to diagonal grid cells
        spts, _ = self.surface_points()
        self.center_ = 0.5 * (np.min(spts, axis=0) + np.max(spts, axis=0))
        self.points_buf_ = np.zeros([Sdf3D.num_interpolants, 3], dtype=np.int)
        self.coords_buf_ = np.zeros([3, ])
        self.pts_ = None

        # tranform sdf basis to grid (X and Z axes are flipped!)
        t_world_grid = self.resolution_ * self.center_
        s_world_grid = 1.0 / self.resolution_

        # FIXME: Since in autolab_core==0.0.4, it applies (un)scale transformation before translation in SimilarityTransform
        # here we shoule use unscaled origin to get the correct world coordinates
        # PS: in world coordinate, the origin here is the left-bottom-down corner of the padded bounding squre box
        t_grid_sdf = self.origin
        self.T_grid_sdf_ = SimilarityTransform(translation=t_grid_sdf,
                                               scale=self.resolution,
                                               from_frame='grid',
                                               to_frame='sdf')
        self.T_sdf_world_ = T_sdf_world
        self.T_grid_world_ = self.T_sdf_world_ * self.T_grid_sdf_

        self.T_sdf_grid_ = self.T_grid_sdf_.inverse()
        self.T_world_grid_ = self.T_grid_world_.inverse()
        self.T_world_sdf_ = self.T_sdf_world_.inverse()

        # optionally use only the absolute values (useful for non-closed meshes in 3D)
        self.use_abs_ = use_abs
        if use_abs:
            self.data_ = np.abs(self.data_)

        self._compute_gradients()
        self.surface_points_w_ = self.transform_pt_grid_to_obj(self.surface_points_.T).T
        surface, _ = self.surface_points(grid_basis=True)
        self.surface_for_signed_val = surface[np.random.choice(len(surface), 1000)]  # FIXME: for speed

    def transform(self, delta_T):
        """ Creates a new SDF with a given pose with respect to world coordinates.

        Parameters
        ----------
        delta_T : :obj:`autolab_core.RigidTransform`
            transform from cur sdf to transformed sdf coords
        """
        new_T_sdf_world = self.T_sdf_world_ * delta_T.inverse().as_frames('sdf', 'sdf')
        return Sdf3D(self.data_, self.origin_, self.resolution_, use_abs=self.use_abs_,
                     T_sdf_world=new_T_sdf_world)

    def _signed_distance(self, coords):
        """Returns the signed distance at the given coordinates, interpolating
        if necessary.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The signed distance at the given coords (interpolated).

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional')
        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')
            # find cloest surface point
            surface = self.surface_for_signed_val
            closest_surface_coord = surface[np.argmin(np.linalg.norm(surface - coords, axis=-1))]
            sd = np.linalg.norm(self.transform_pt_grid_to_obj(closest_surface_coord) -
                                self.transform_pt_grid_to_obj(coords)) + \
                                self.data_[closest_surface_coord[0], closest_surface_coord[1], closest_surface_coord[2]]
        else:
            # snap to grid dims
            self.coords_buf_[0] = max(0, min(coords[0], self.dims_[0] - 1))
            self.coords_buf_[1] = max(0, min(coords[1], self.dims_[1] - 1))
            self.coords_buf_[2] = max(0, min(coords[2], self.dims_[2] - 1))
            # regular indexing if integers
            if np.issubdtype(type(coords[0]), np.integer) and \
               np.issubdtype(type(coords[1]), np.integer) and \
               np.issubdtype(type(coords[2]), np.integer):
                return self.data_[int(self.coords_buf_[0]), int(self.coords_buf_[1]), int(self.coords_buf_[2])]

            # otherwise interpolate
            min_coords = np.floor(self.coords_buf_)
            max_coords = min_coords + 1  # assumed to be on grid
            self.points_buf_[Sdf3D.min_coords_x, 0] = min_coords[0]
            self.points_buf_[Sdf3D.max_coords_x, 0] = max_coords[0]
            self.points_buf_[Sdf3D.min_coords_y, 1] = min_coords[1]
            self.points_buf_[Sdf3D.max_coords_y, 1] = max_coords[1]
            self.points_buf_[Sdf3D.min_coords_z, 2] = min_coords[2]
            self.points_buf_[Sdf3D.max_coords_z, 2] = max_coords[2]

            # bilinearly interpolate points
            sd = 0.0
            for i in range(Sdf3D.num_interpolants):
                p = self.points_buf_[i, :]
                if self.is_out_of_bounds(p):
                    v = 0.0
                else:
                    v = self.data_[p[0], p[1], p[2]]
                w = np.prod(-np.abs(p - self.coords_buf_) + 1)
                sd = sd + w * v

        return sd

    def __getitem__(self, coords):
        """Returns the signed distance at the given coordinates.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The signed distance at the given coords (interpolated).

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        return self._signed_distance(coords)

    def gradient(self, coords):
        """Returns the SDF gradient at the given coordinates, interpolating if necessary

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The gradient at the given coords (interpolated).

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional')

        # log warning if out of bounds access
        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        self.coords_buf_[0] = max(0, min(coords[0], self.dims_[0] - 1))
        self.coords_buf_[1] = max(0, min(coords[1], self.dims_[1] - 1))
        self.coords_buf_[2] = max(0, min(coords[2], self.dims_[2] - 1))

        # regular indexing if integers
        if type(coords[0]) is int and type(coords[1]) is int and type(coords[2]) is int:
            self.coords_buf_ = self.coords_buf_.astype(np.int)
            return self.data_[self.coords_buf_[0], self.coords_buf_[1], self.coords_buf_[2]]

        # otherwise interpolate
        min_coords = np.floor(self.coords_buf_)
        max_coords = min_coords + 1
        self.points_buf_[Sdf3D.min_coords_x, 0] = min_coords[0]
        self.points_buf_[Sdf3D.max_coords_x, 0] = min_coords[0]
        self.points_buf_[Sdf3D.min_coords_y, 1] = min_coords[1]
        self.points_buf_[Sdf3D.max_coords_y, 1] = max_coords[1]
        self.points_buf_[Sdf3D.min_coords_z, 2] = min_coords[2]
        self.points_buf_[Sdf3D.max_coords_z, 2] = max_coords[2]

        # bilinear interpolation
        g = np.zeros(3)
        gp = np.zeros(3)
        w_sum = 0.0
        for i in range(Sdf3D.num_interpolants):
            p = self.points_buf_[i, :]
            if self.is_out_of_bounds(p):
                gp[0] = 0.0
                gp[1] = 0.0
                gp[2] = 0.0
            else:
                gp[0] = self.gradients_[0][p[0], p[1], p[2]]
                gp[1] = self.gradients_[1][p[0], p[1], p[2]]
                gp[2] = self.gradients_[2][p[0], p[1], p[2]]

            w = np.prod(-np.abs(p - self.coords_buf_) + 1)
            g = g + w * gp

        return g

    def curvature(self, coords, delta=0.001):
        """
        Returns an approximation to the local SDF curvature (Hessian) at the
        given coordinate in grid basis.

        Parameters
        ---------
        coords : numpy 3-vector
            the grid coordinates at which to get the curvature
        delta :
        Returns
        -------
        curvature : 3x3 ndarray of the curvature at the surface points
        """
        # perturb local coords
        coords_x_up = coords + np.array([delta, 0, 0])
        coords_x_down = coords + np.array([-delta, 0, 0])
        coords_y_up = coords + np.array([0, delta, 0])
        coords_y_down = coords + np.array([0, -delta, 0])
        coords_z_up = coords + np.array([0, 0, delta])
        coords_z_down = coords + np.array([0, 0, -delta])

        # get gradient
        grad_x_up = self.gradient(coords_x_up)
        grad_x_down = self.gradient(coords_x_down)
        grad_y_up = self.gradient(coords_y_up)
        grad_y_down = self.gradient(coords_y_down)
        grad_z_up = self.gradient(coords_z_up)
        grad_z_down = self.gradient(coords_z_down)

        # finite differences
        curvature_x = (grad_x_up - grad_x_down) / (4 * delta)
        curvature_y = (grad_y_up - grad_y_down) / (4 * delta)
        curvature_z = (grad_z_up - grad_z_down) / (4 * delta)
        curvature = np.c_[curvature_x, np.c_[curvature_y, curvature_z]]
        curvature = curvature + curvature.T
        return curvature

    def surface_normal(self, coords, delta=1.5):
        """Returns the sdf surface normal at the given coordinates by
        computing the tangent plane using SDF interpolation.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        delta : float
            A radius for collecting surface points near the target coords
            for calculating the surface normal.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3-dimensional ndarray that represents the surface normal.

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional')

        # log warning if out of bounds access
        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        # coords[0] = max(0, min(coords[0], self.dims_[0] - 1))
        # coords[1] = max(0, min(coords[1], self.dims_[1] - 1))
        # coords[2] = max(0, min(coords[2], self.dims_[2] - 1))
        index_coords = np.zeros(3)

        # check points on surface
        sdf_val = self[coords]
        if np.abs(sdf_val) >= self.surface_thresh_:
            logging.debug('Cannot compute normal. Point must be on surface')
            return None

        # collect all surface points within the delta sphere
        X = []
        d = np.zeros(3)
        dx = -delta
        while dx <= delta:
            dy = -delta
            while dy <= delta:
                dz = -delta
                while dz <= delta:
                    d = np.array([dx, dy, dz])
                    if dx != 0 or dy != 0 or dz != 0:
                        d = delta * d / np.linalg.norm(d)
                    index_coords[0] = coords[0] + d[0]
                    index_coords[1] = coords[1] + d[1]
                    index_coords[2] = coords[2] + d[2]
                    sdf_val = self[index_coords]
                    if np.abs(sdf_val) < self.surface_thresh_:
                        X.append([index_coords[0], index_coords[1], index_coords[2], sdf_val])
                    dz += delta
                dy += delta
            dx += delta

        # fit a plane to the surface points
        X.sort(key=lambda x: x[3])
        X = np.array(X)[:, :3]
        A = X - np.mean(X, axis=0)
        try:
            U, S, V = np.linalg.svd(A.T)
            n = U[:, 2]
        except:
            logging.warning('Tangent plane does not exist. Returning None.')
            return None
        # make sure surface normal is outward
        # referenced from Zhou Xian's github, but if the model is not watertight, this method may fail
        # https://github.com/zhouxian/meshpy_berkeley/commit/96428f3b7af618a0828a7eb88f22541cdafacfc7
        if self[coords + n * 0.01] < self[coords]:
            n = -n
        return n

    def _compute_surface_points(self):
        surface_points = np.where(np.abs(self.data_) < self.surface_thresh_)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        self.surface_points_ = np.c_[x, np.c_[y, z]]
        self.surface_vals_ = self.data_[self.surface_points_[:, 0], self.surface_points_[:, 1],
                                        self.surface_points_[:, 2]]

    def surface_points(self, grid_basis=True):
        """Returns the points on the surface.

        Parameters
        ----------
        grid_basis : bool
            If False, the surface points are transformed to the world frame.
            If True (default), the surface points are left in grid coordinates.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of int, :obj:`numpy.ndarray` of float
            The points on the surface and the signed distances at those points.
        """
        if not grid_basis:
            return self.surface_points_w_, self.surface_vals_
        return self.surface_points_, self.surface_vals_

    def rescale(self, scale):
        """ Rescale an SDF by a given scale factor.

        Parameters
        ----------
        scale : float
            the amount to scale the SDF

        Returns
        -------
        :obj:`Sdf3D`
            new sdf with given scale
        """
        resolution_tf = scale * self.resolution_
        return Sdf3D(self.data_, self.origin_, resolution_tf, use_abs=self.use_abs_,
                     T_sdf_world=self.T_sdf_world_)

    def transform_dense(self, delta_T, detailed=False):
        """ Transform the grid by pose T and scale with canonical reference
        frame at the SDF center with axis alignment.

        Parameters
        ----------
        delta_T : SimilarityTransform
            the transformation from the current frame of reference to the new frame of reference
        detailed : bool
            whether or not to use interpolation

        Returns
        -------
        :obj:`Sdf3D`
            new sdf with grid warped by T
        """
        # map all surface points to their new location
        start_t = time.clock()

        # form points array
        if self.pts_ is None:
            [x_ind, y_ind, z_ind] = np.indices(self.dims_)
            self.pts_ = np.c_[x_ind.flatten().T, np.c_[y_ind.flatten().T, z_ind.flatten().T]].astype(np.float32)

        # transform points
        num_pts = self.pts_.shape[0]
        pts_sdf = self.T_grid_sdf_ * PointCloud(self.pts_.T, frame='grid')
        pts_sdf_tf = delta_T.as_frames('sdf', 'sdf') * pts_sdf
        pts_grid_tf = self.T_sdf_grid_ * pts_sdf_tf
        pts_tf = pts_grid_tf.data.T
        all_points_t = time.clock()

        # transform the center
        origin_sdf = self.T_grid_sdf_ * Point(self.origin_, frame='grid')
        origin_sdf_tf = delta_T.as_frames('sdf', 'sdf') * origin_sdf
        origin_tf = self.T_sdf_grid_ * origin_sdf_tf
        origin_tf = origin_tf.data

        # use same resolution (since indices will be rescaled)
        resolution_tf = self.resolution_
        origin_res_t = time.clock()

        # add each point to the new pose
        if detailed:
            sdf_data_tf = np.zeros([num_pts, 1])
            for i in range(num_pts):
                sdf_data_tf[i] = self[pts_tf[i, 0], pts_tf[i, 1], pts_tf[i, 2]]
        else:
            pts_tf_round = np.round(pts_tf).astype(np.int64)

            # snap to closest boundary
            pts_tf_round[:, 0] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:, 0]], axis=1)
            pts_tf_round[:, 0] = np.min(np.c_[(self.dims_[0] - 1) * np.ones([num_pts, 1]), pts_tf_round[:, 0]], axis=1)

            pts_tf_round[:, 1] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:, 1]], axis=1)
            pts_tf_round[:, 1] = np.min(np.c_[(self.dims_[1] - 1) * np.ones([num_pts, 1]), pts_tf_round[:, 1]], axis=1)

            pts_tf_round[:, 2] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:, 2]], axis=1)
            pts_tf_round[:, 2] = np.min(np.c_[(self.dims_[2] - 1) * np.ones([num_pts, 1]), pts_tf_round[:, 2]], axis=1)

            sdf_data_tf = self.data_[pts_tf_round[:, 0], pts_tf_round[:, 1], pts_tf_round[:, 2]]

        sdf_data_tf_grid = sdf_data_tf.reshape(self.dims_)
        tf_t = time.clock()

        logging.debug('Sdf3D: Time to transform coords: %f' % (all_points_t - start_t))
        logging.debug('Sdf3D: Time to transform origin: %f' % (origin_res_t - all_points_t))
        logging.debug('Sdf3D: Time to transfer sd: %f' % (tf_t - origin_res_t))
        return Sdf3D(sdf_data_tf_grid, origin_tf, resolution_tf, use_abs=self._use_abs_, T_sdf_world=self.T_sdf_world_)

    def transform_pt_obj_to_grid(self, x_sdf, direction=False):
        """ Converts a point in sdf coords to the grid basis. If direction then don't translate.

        Parameters
        ----------
        x_sdf : numpy 3xN ndarray or numeric scalar
            points to transform from sdf basis in meters to grid basis
        direction : bool
        Returns
        -------
        x_grid : numpy 3xN ndarray or scalar
            points in grid basis
        """
        if isinstance(x_sdf, Number):
            return self.T_world_grid_.scale * x_sdf
        if direction:
            points_sdf = NormalCloud(x_sdf.astype(np.float32), frame='world')
        else:
            points_sdf = PointCloud(x_sdf.astype(np.float32), frame='world')
        x_grid = self.T_world_grid_ * points_sdf
        return x_grid.data

    def transform_pt_grid_to_obj(self, x_grid, direction=False):
        """ Converts a point in grid coords to the world basis. If direction then don't translate.
        
        Parameters
        ----------
        x_grid : numpy 3xN ndarray or numeric scalar
            points to transform from grid basis to sdf basis in meters
        direction : bool
        Returns
        -------
        x_sdf : numpy 3xN ndarray
            points in sdf basis (meters)
        """
        if isinstance(x_grid, Number):
            return self.T_grid_world_.scale * x_grid
        if direction:
            points_grid = NormalCloud(x_grid.astype(np.float32), frame='grid')
        else:
            points_grid = PointCloud(x_grid.astype(np.float32), frame='grid')
        x_sdf = self.T_grid_world_ * points_grid
        return x_sdf.data

    @staticmethod
    def find_zero_crossing_linear(x1, y1, x2, y2):
        """ Find zero crossing using linear approximation"""
        # NOTE: use sparingly, approximations can be shoddy
        d = x2 - x1
        t1 = 0
        t2 = np.linalg.norm(d)
        v = d / t2

        m = (y2 - y1) / (t2 - t1)
        b = y1
        t_zc = -b / m
        x_zc = x1 + t_zc * v
        return x_zc

    @staticmethod
    def find_zero_crossing_quadratic(x1, y1, x2, y2, x3, y3, eps=1.0):
        """ Find zero crossing using quadratic approximation along 1d line"""
        # compute coords along 1d line
        v = x2 - x1
        v = v / np.linalg.norm(v)
        if v[v != 0].shape[0] == 0:
            logging.error('Difference is 0. Probably a bug')

        t1 = 0
        t2 = (x2 - x1)[v != 0] / v[v != 0]
        t2 = t2[0]
        t3 = (x3 - x1)[v != 0] / v[v != 0]
        t3 = t3[0]

        # solve for quad approx
        x1_row = np.array([t1 ** 2, t1, 1])
        x2_row = np.array([t2 ** 2, t2, 1])
        x3_row = np.array([t3 ** 2, t3, 1])
        X = np.array([x1_row, x2_row, x3_row])
        y_vec = np.array([y1, y2, y3])
        try:
            w = np.linalg.solve(X, y_vec)
        except np.linalg.LinAlgError:
            logging.error('Singular matrix. Probably a bug')
            return None

        # get positive roots
        possible_t = np.roots(w)
        t_zc = None
        for i in range(possible_t.shape[0]):
            if 0 <= possible_t[i] <= 10 and not np.iscomplex(possible_t[i]):
                t_zc = possible_t[i]

        # if no positive roots find min
        if np.abs(w[0]) < 1e-10:
            return None

        if t_zc is None:
            t_zc = -w[1] / (2 * w[0])

        if t_zc < -eps or t_zc > eps:
            return None

        x_zc = x1 + t_zc * v
        return x_zc
