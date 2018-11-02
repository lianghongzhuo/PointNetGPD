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
Contact class that encapsulates friction cone and surface window computation.
Authors: Brian Hou and Jeff Mahler
"""

from abc import ABCMeta, abstractmethod
import itertools as it
import logging
import numpy as np
from skimage.restoration import denoise_bilateral

from autolab_core import RigidTransform

from dexnet.constants import NO_CONTACT_DIST
from dexnet.constants import WIN_DIST_LIM

import IPython
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# class Contact(metaclass=ABCMeta):  # for python3
class Contact:
    """ Abstract class for contact models. """
    __metaclass__ = ABCMeta

class Contact3D(Contact):
    """ 3D contact points.

    Attributes
    ----------
    graspable : :obj:`GraspableObject3D`
        object to use to get contact information
    contact_point : 3x1 :obj:`numpy.ndarray`
        point of contact on the object
    in_direction : 3x1 :obj:`numpy.ndarray`
        direction along which contact was made
    normal : normalized 3x1 :obj:`numpy.ndarray`
        surface normal at the contact point
    """

    def __init__(self, graspable, contact_point, in_direction=None):
        self.graspable_ = graspable
        self.point_ = contact_point  # in world coordinates

        # cached attributes
        self.in_direction_ = in_direction  # inward facing grasp axis
        self.friction_cone_ = None
        self.normal_ = None  # outward facing normal
        self.surface_info_ = None

        self._compute_normal()

    @property
    def graspable(self):
        return self.graspable_

    @property
    def point(self):
        return self.point_

    @property
    def normal(self):
        return self.normal_

    @normal.setter
    def normal(self, normal):
        self.normal_ = normal

    @property
    def in_direction(self):
        return self.in_direction_

    def _compute_normal(self):
        """Compute outward facing normal at contact, according to in_direction.
        Indexes into the SDF grid coordinates to lookup the normal info.
        """
        # tf to grid
        as_grid = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        on_surface, _ = self.graspable.sdf.on_surface(as_grid)
        if not on_surface:
            logging.debug('Contact point not on surface')
            return None

        # compute outward facing normal from SDF
        normal = self.graspable.sdf.surface_normal(as_grid)

        # flip normal to point outward if in_direction is defined
        if self.in_direction_ is not None and np.dot(self.in_direction_, normal) > 0:
            normal = -normal

        # transform to world frame
        normal = self.graspable.sdf.transform_pt_grid_to_obj(normal, direction=True)
        self.normal_ = normal

    def tangents(self, direction=None, align_axes=True, max_samples=1000):
        """Returns the direction vector and tangent vectors at a contact point.
        The direction vector defaults to the *inward-facing* normal vector at
        this contact.
        The direction and tangent vectors for a right handed coordinate frame.

        Parameters
        ----------
        direction : 3x1 :obj:`numpy.ndarray`
            direction to find orthogonal plane for
        align_axes : bool
            whether or not to align the tangent plane to the object reference frame
        max_samples : int
            number of samples to use in discrete optimization for alignment of reference frame

        Returns
        -------
        direction : normalized 3x1 :obj:`numpy.ndarray`
            direction to find orthogonal plane for
        t1 : normalized 3x1 :obj:`numpy.ndarray`
            first tangent vector, x axis
        t2 : normalized 3x1 :obj:`numpy.ndarray`
            second tangent vector, y axis
        """
        # illegal contact, cannot return tangents
        if self.normal_ is None:
            return None, None, None

        # default to inward pointing normal
        if direction is None:
            direction = -self.normal_

        # force direction to face inward
        if np.dot(self.normal_, direction) > 0:
            direction = -direction

        # transform to 
        direction = direction.reshape((3, 1))  # make 2D for SVD

        # get orthogonal plane
        U, _, _ = np.linalg.svd(direction)

        # U[:, 1:] spans the tanget plane at the contact
        x, y = U[:, 1], U[:, 2]

        # make sure t1 and t2 obey right hand rule
        z_hat = np.cross(x, y)
        if z_hat.dot(direction) < 0:
            y = -y
        v = x
        w = y

        # redefine tangent x axis to automatically align with the object x axis
        if align_axes:
            max_ip = 0
            max_theta = 0
            target = np.array([1, 0, 0])
            theta = 0
            d_theta = 2 * np.pi / float(max_samples)
            for i in range(max_samples):
                v = np.cos(theta) * x + np.sin(theta) * y
                if v.dot(target) > max_ip:
                    max_ip = v.dot(target)
                    max_theta = theta
                theta = theta + d_theta

            v = np.cos(max_theta) * x + np.sin(max_theta) * y
            w = np.cross(direction.ravel(), v)
        return np.squeeze(direction), v, w

    def reference_frame(self, align_axes=True):
        """Returns the local reference frame of the contact.
        Z axis in the in direction (or surface normal if not specified)
        X and Y axes in the tangent plane to the direction

        Parameters
        ----------
        align_axes : bool
            whether or not to align to the object axes

        Returns
        -------
        :obj:`RigidTransform`
            rigid transformation from contact frame to object frame
        """
        t_obj_contact = self.point
        rz, rx, ry = self.tangents(self.in_direction_, align_axes=align_axes)
        R_obj_contact = np.array([rx, ry, rz]).T
        T_contact_obj = RigidTransform(rotation=R_obj_contact,
                                       translation=t_obj_contact,
                                       from_frame='contact', to_frame='obj')
        return T_contact_obj

    def normal_force_magnitude(self):
        """ Returns the component of the force that the contact would apply along the normal direction.

        Returns
        -------
        float
            magnitude of force along object surface normal
        """
        normal_force_mag = 1.0
        if self.in_direction_ is not None and self.normal_ is not None:
            in_normal = -self.normal_
            in_direction_norm = self.in_direction_ / np.linalg.norm(self.in_direction_)
            normal_force_mag = np.dot(in_direction_norm, in_normal)
        return max(normal_force_mag, 0.0)

    def friction_cone(self, num_cone_faces=8, friction_coef=0.5):
        """ Computes the friction cone and normal for a contact point.

        Parameters
        ----------
        num_cone_faces : int
            number of cone faces to use in discretization
        friction_coef : float 
            coefficient of friction at contact point
        
        Returns
        -------
        success : bool
            False when cone can't be computed
        cone_support : :obj:`numpy.ndarray`
            array where each column is a vector on the boundary of the cone
        normal : normalized 3x1 :obj:`numpy.ndarray`
            outward facing surface normal
        """
        if self.friction_cone_ is not None and self.normal_ is not None:
            return True, self.friction_cone_, self.normal_

        # get normal and tangents
        in_normal, t1, t2 = self.tangents()
        if in_normal is None:
            return False, self.friction_cone_, self.normal_

        friction_cone_valid = True

        # check whether contact would slip, which is whether or not the tangent force is always
        # greater than the frictional force
        if self.in_direction_ is not None:
            in_direction_norm = self.in_direction_ / np.linalg.norm(self.in_direction_)
            normal_force_mag = self.normal_force_magnitude()
            tan_force_x = np.dot(in_direction_norm, t1)
            tan_force_y = np.dot(in_direction_norm, t2)
            tan_force_mag = np.sqrt(tan_force_x ** 2 + tan_force_y ** 2)
            friction_force_mag = friction_coef * normal_force_mag

            if friction_force_mag < tan_force_mag:
                logging.debug('Contact would slip')
                return False, self.friction_cone_, self.normal_

        # set up friction cone
        tan_len = friction_coef
        force = in_normal
        cone_support = np.zeros((3, num_cone_faces))

        # find convex combinations of tangent vectors
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + t2 * np.sin(
                2 * np.pi * (float(j) / num_cone_faces))
            cone_support[:, j] = force + friction_coef * tan_vec

        self.friction_cone_ = cone_support
        return True, self.friction_cone_, self.normal_

    def torques(self, forces):
        """
        Get the torques that can be applied by a set of force vectors at the contact point.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            the forces applied at the contact

        Returns
        -------
        success : bool
            whether or not computation was successful
        torques : 3xN :obj:`numpy.ndarray`
            the torques that can be applied by given forces at the contact
        """
        as_grid = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        on_surface, _ = self.graspable.sdf.on_surface(as_grid)
        if not on_surface:
            logging.debug('Contact point not on surface')
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = self.graspable.moment_arm(self.point)
        for i in range(num_forces):
            torques[:, i] = np.cross(moment_arm, forces[:, i])

        return True, torques

    def surface_window_sdf(self, width=1e-2, num_steps=21):
        """Returns a window of SDF values on the tangent plane at a contact point.
        Used for patch computation.

        Parameters
        ----------
        width : float
            width of the window in obj frame
        num_steps : int
            number of steps to use along the contact in direction

        Returns
        -------
        window : NUM_STEPSxNUM_STEPS :obj:`numpy.ndarray`
            array of distances from tangent plane to obj along in direction, False if surface window can't be computed
        """
        in_normal, t1, t2 = self.tangents()
        if in_normal is None:  # normal and tangents not found
            return False

        scales = np.linspace(-width / 2.0, width / 2.0, num_steps)
        window = np.zeros(num_steps ** 2)
        for i, (c1, c2) in enumerate(it.product(scales, repeat=2)):
            curr_loc = self.point + c1 * t1 + c2 * t2
            curr_loc_grid = self.graspable.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.graspable.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = -1e-2
                continue

            window[i] = self.graspable.sdf[curr_loc_grid]
        return window.reshape((num_steps, num_steps))

    def _compute_surface_window_projection(self, u1=None, u2=None, width=1e-2,
                                           num_steps=21, max_projection=0.1, back_up=0, samples_per_grid=2.0,
                                           sigma_range=0.1, sigma_spatial=1, direction=None, vis=False,
                                           compute_weighted_covariance=False,
                                           disc=False, num_radial_steps=5, debug_objs=None):
        """Compute the projection window onto the basis defined by u1 and u2.
        Params:
            u1, u2 - orthogonal numpy 3 arrays

            width - float width of the window in obj frame
            num_steps - int number of steps
            max_projection - float maximum amount to search forward for a
                contact (meters)

            back_up - amount in meters to back up before projecting
            samples_per_grid - float number of samples per grid when finding contacts
            sigma - bandwidth of gaussian filter on window
            direction - dir to do the projection along
            compute_weighted_covariance - whether to return the weighted
               covariance matrix, along with the window
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        direction, t1, t2 = self.tangents(direction)
        if direction is None:  # normal and tangents not found
            raise ValueError('Direction could not be computed')
        if u1 is not None and u2 is not None:  # use given basis
            t1, t2 = u1, u2

        # number of samples used when looking for contacts
        no_contact = NO_CONTACT_DIST
        num_samples = int(samples_per_grid * (max_projection + back_up) / self.graspable.sdf.resolution)
        window = np.zeros(num_steps ** 2)

        res = width / num_steps
        scales = np.linspace(-width / 2.0 + res / 2.0, width / 2.0 - res / 2.0, num_steps)
        scales_it = it.product(scales, repeat=2)
        if disc:
            scales_it = []
            for i in range(num_steps):
                theta = 2.0 * np.pi / i
                for j in range(num_radial_steps):
                    r = (j + 1) * width / num_radial_steps
                    p = (r * np.cos(theta), r * np.sin(theta))
                    scales_it.append(p)

        # start computing weighted covariance matrix
        if compute_weighted_covariance:
            cov = np.zeros((3, 3))
            cov_weight = 0

        if vis:
            ax = plt.gca(projection='3d')
            self.graspable_.sdf.scatter()

        for i, (c1, c2) in enumerate(scales_it):
            curr_loc = self.point + c1 * t1 + c2 * t2
            curr_loc_grid = self.graspable.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.graspable.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = no_contact
                continue

            if vis:
                ax.scatter(curr_loc_grid[0], curr_loc_grid[1], curr_loc_grid[2], s=130, c='y')

            found, projection_contact = self.graspable._find_projection(
                curr_loc, direction, max_projection, back_up, num_samples, vis=vis)

            if found:
                # logging.debug('%d found.' %(i))
                sign = direction.dot(projection_contact.point - curr_loc)
                projection = (sign / abs(sign)) * np.linalg.norm(projection_contact.point - curr_loc)
                projection = min(projection, max_projection)

                if compute_weighted_covariance:
                    # weight according to SHOT: R - d_i
                    weight = width / np.sqrt(2) - np.sqrt(c1 ** 2 + c2 ** 2)
                    diff = (projection_contact.point - self.point).reshape((3, 1))
                    cov += weight * np.dot(diff, diff.T)
                    cov_weight += weight
            else:
                logging.debug('%d not found.' % (i))
                projection = no_contact

            window[i] = projection

        if vis:
            plt.show()

        if not disc:
            window = window.reshape((num_steps, num_steps)).T  # transpose to make x-axis along columns
            if debug_objs is not None:
                debug_objs.append(window)
            # apply bilateral filter
            if sigma_range > 0.0 and sigma_spatial > 0.0:
                window_min_val = np.min(window)
                window_pos = window - window_min_val
                window_pos_blur = denoise_bilateral(window_pos, sigma_range=sigma_range, sigma_spatial=sigma_spatial,
                                                    mode='nearest')
                window = window_pos_blur + window_min_val
            if compute_weighted_covariance:
                if cov_weight > 0:
                    return window, cov / cov_weight
                return window, cov
        return window

    def surface_window_projection_unaligned(self, width=1e-2, num_steps=21,
                                            max_projection=0.1, back_up=0.0, samples_per_grid=2.0,
                                            sigma=1.5, direction=None, vis=False):
        """Projects the local surface onto the tangent plane at a contact point. Deprecated.
        """
        return self._compute_surface_window_projection(width=width,
                                                       num_steps=num_steps, max_projection=max_projection,
                                                       back_up=back_up, samples_per_grid=samples_per_grid,
                                                       sigma=sigma, direction=direction, vis=vis)

    def surface_window_projection(self, width=1e-2, num_steps=21,
                                  max_projection=0.1, back_up=0.0, samples_per_grid=2.0,
                                  sigma_range=0.1, sigma_spatial=1, direction=None, compute_pca=False, vis=False,
                                  debug_objs=None):
        """Projects the local surface onto the tangent plane at a contact point.

        Parameters
        ----------
        width : float
            width of the window in obj frame
        num_steps : int 
            number of steps to use along the in direction
        max_projection : float
            maximum amount to search forward for a contact (meters)
        back_up : float
            amount to back up before finding a contact in meters
        samples_per_grid : float
            number of samples per grid when finding contacts
        sigma_range : float
            bandwidth of bilateral range filter on window
        sigma_spatial : float
            bandwidth of gaussian spatial filter of bilateral filter
        direction : 3x1 :obj:`numpy.ndarray`
            dir to do the projection along

        Returns
        -------
        window : NUM_STEPSxNUM_STEPS :obj:`numpy.ndarray`
            array of distances from tangent plane to obj, False if surface window can't be computed
        """
        # get initial projection
        direction, t1, t2 = self.tangents(direction)
        window, cov = self._compute_surface_window_projection(t1, t2,
                                                              width=width, num_steps=num_steps,
                                                              max_projection=max_projection,
                                                              back_up=back_up, samples_per_grid=samples_per_grid,
                                                              sigma_range=sigma_range, sigma_spatial=sigma_spatial,
                                                              direction=direction,
                                                              vis=False, compute_weighted_covariance=True,
                                                              debug_objs=debug_objs)

        if not compute_pca:
            return window

        # compute principal axis
        pca = PCA()
        pca.fit(cov)
        R = pca.components_
        principal_axis = R[0, :]
        if np.isclose(abs(np.dot(principal_axis, direction)), 1):
            # principal axis is aligned with direction of projection, use secondary axis
            principal_axis = R[1, :]

        if vis:
            # reshape window
            window = window.reshape((num_steps, num_steps))

            # project principal axis onto tangent plane (t1, t2) to get u1
            u1t = np.array([np.dot(principal_axis, t1), np.dot(principal_axis, t2)])
            u2t = np.array([-u1t[1], u1t[0]])
            if sigma > 0:
                window = spfilt.gaussian_filter(window, sigma)
            plt.figure()
            plt.title('Principal Axis')
            plt.imshow(window, extent=[0, num_steps - 1, num_steps - 1, 0],
                       interpolation='none', cmap=plt.cm.binary)
            plt.colorbar()
            plt.clim(-WIN_DIST_LIM, WIN_DIST_LIM)  # fixing color range for visual comparisons
            center = num_steps // 2
            plt.scatter([center, center * u1t[0] + center], [center, -center * u1t[1] + center], color='blue')
            plt.scatter([center, center * u2t[0] + center], [center, -center * u2t[1] + center], color='green')

        u1 = np.dot(principal_axis, t1) * t1 + np.dot(principal_axis, t2) * t2
        u2 = np.cross(direction, u1)  # u2 must be orthogonal to u1 on plane
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)

        window = self._compute_surface_window_projection(u1, u2,
                                                         width=width, num_steps=num_steps,
                                                         max_projection=max_projection,
                                                         back_up=back_up, samples_per_grid=samples_per_grid,
                                                         sigma=sigma, direction=direction, vis=False)

        # arbitrarily require that right_avg > left_avg (inspired by SHOT)
        left_avg = np.average(window[:, :num_steps // 2])
        right_avg = np.average(window[:, num_steps // 2:])
        if left_avg > right_avg:
            # need to flip both u1 and u2, i.e. rotate 180 degrees
            window = np.rot90(window, k=2)

        if vis:
            if sigma > 0:
                window = spfilt.gaussian_filter(window, sigma)
            plt.figure()
            plt.title('Tfd')
            plt.imshow(window, extent=[0, num_steps - 1, num_steps - 1, 0],
                       interpolation='none', cmap=plt.cm.binary)
            plt.colorbar()
            plt.clim(-WIN_DIST_LIM, WIN_DIST_LIM)  # fixing color range for visual comparisons
            plt.show()

        return window

    def surface_information(self, width, num_steps, sigma_range=0.1, sigma_spatial=1,
                            back_up=0.0, max_projection=0.1, direction=None, debug_objs=None, samples_per_grid=2):
        """
        Returns the local surface window, gradient, and curvature for a single contact.

        Parameters
        ----------
        width : float
            width of surface window in object frame
        num_steps : int 
            number of steps to use along the in direction
        sigma_range : float
            bandwidth of bilateral range filter on window
        sigma_spatial : float
            bandwidth of gaussian spatial filter of bilateral filter
        back_up : float
            amount to back up before finding a contact in meters
        max_projection : float
            maximum amount to search forward for a contact (meters)
        direction : 3x1 :obj:`numpy.ndarray`
            direction along width to render the window
        debug_objs : :obj:`list`
            list to put debugging info into
        samples_per_grid : float
            number of samples per grid when finding contacts
        
        Returns
        -------
        surface_window : :obj:`SurfaceWindow`
            window information for local surface patch of contact on the given object
        """
        if self.surface_info_ is not None:
            return self.surface_info_

        if direction is None:
            direction = self.in_direction_

        proj_window = self.surface_window_projection(width, num_steps,
                                                     sigma_range=sigma_range, sigma_spatial=sigma_spatial,
                                                     back_up=back_up, max_projection=max_projection,
                                                     samples_per_grid=samples_per_grid,
                                                     direction=direction, vis=False, debug_objs=debug_objs)

        if proj_window is None:
            raise ValueError('Surface window could not be computed')

        grad_win = np.gradient(proj_window)
        hess_x = np.gradient(grad_win[0])
        hess_y = np.gradient(grad_win[1])

        gauss_curvature = np.zeros(proj_window.shape)
        for i in range(num_steps):
            for j in range(num_steps):
                local_hess = np.array([[hess_x[0][i, j], hess_x[1][i, j]],
                                       [hess_y[0][i, j], hess_y[1][i, j]]])
                # symmetrize
                local_hess = (local_hess + local_hess.T) / 2.0
                # curvature
                gauss_curvature[i, j] = np.linalg.det(local_hess)

        return SurfaceWindow(proj_window, grad_win, hess_x, hess_y, gauss_curvature)

    def plot_friction_cone(self, color='y', scale=1.0):
        success, cone, in_normal = self.friction_cone()

        ax = plt.gca(projection='3d')
        self.graspable.sdf.scatter()  # object
        x, y, z = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        nx, ny, nz = self.graspable.sdf.transform_pt_obj_to_grid(in_normal, direction=True)
        ax.scatter([x], [y], [z], c=color, s=60)  # contact
        ax.scatter([x - nx], [y - ny], [z - nz], c=color, s=60)  # normal
        if success:
            ax.scatter(x + scale * cone[0], y + scale * cone[1], z + scale * cone[2], c=color, s=40)  # cone

        ax.set_xlim3d(0, self.graspable.sdf.dims_[0])
        ax.set_ylim3d(0, self.graspable.sdf.dims_[1])
        ax.set_zlim3d(0, self.graspable.sdf.dims_[2])

        return plt.Rectangle((0, 0), 1, 1, fc=color)  # return a proxy for legend


class SurfaceWindow:
    """Struct for encapsulating local surface window features.

    Attributes
    ----------
    proj_win : NxN :obj:`numpy.ndarray`
        the window of distances to a surface (depth image created by orthographic projection)
    grad : NxN :obj:`numpy.ndarray`
        X and Y gradients of the projection window
    hess_x : NxN :obj:`numpy.ndarray`
        hessian, partial derivatives of the X gradient window
    hess_y : NxN :obj:`numpy.ndarray`
        hessian, partial derivatives of the Y gradient window
    gauss_curvature : NxN :obj:`numpy.ndarray`
        gauss curvature at each point (function of hessian determinant)
    """

    def __init__(self, proj_win, grad, hess_x, hess_y, gauss_curvature):
        self.proj_win_ = proj_win
        self.grad_ = grad
        self.hess_x_ = hess_x
        self.hess_y_ = hess_y
        self.gauss_curvature_ = gauss_curvature

    @property
    def proj_win_2d(self):
        return self.proj_win_

    @property
    def proj_win(self):
        return self.proj_win_.flatten()

    @property
    def grad_x(self):
        return self.grad_[0].flatten()

    @property
    def grad_y(self):
        return self.grad_[1].flatten()

    @property
    def grad_x_2d(self):
        return self.grad_[0]

    @property
    def grad_y_2d(self):
        return self.grad_[1]

    @property
    def curvature(self):
        return self.gauss_curvature_.flatten()

    def asarray(self, proj_win_weight=0.0, grad_x_weight=0.0,
                grad_y_weight=0.0, curvature_weight=0.0):
        proj_win = proj_win_weight * self.proj_win
        grad_x = grad_x_weight * self.grad_x
        grad_y = grad_y_weight * self.grad_y
        curvature = curvature_weight * self.gauss_curvature
        return np.append([], [proj_win, grad_x, grad_y, curvature])
