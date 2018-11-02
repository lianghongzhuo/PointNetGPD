"""
A basic struct-like Stable Pose class to make accessing pose probability and rotation matrix easier

Author: Matt Matl and Nikhil Sharma
"""
import numpy as np

from autolab_core import RigidTransform

d_theta = np.deg2rad(1)

class StablePose(object):
    """A representation of a mesh's stable pose.

    Attributes
    ----------
    p : float
        Probability associated with this stable pose.
    r : :obj:`numpy.ndarray` of :obj`numpy.ndarray` of float
        3x3 rotation matrix that rotates the mesh into the stable pose from
        standardized coordinates.
    x0 : :obj:`numpy.ndarray` of float
        3D point in the mesh that is resting on the table.
    face : :obj:`numpy.ndarray`
        3D vector of indices corresponding to vertices forming the resting face
    stp_id : :obj:`str`
        A string identifier for the stable pose
    T_obj_table : :obj:`RigidTransform`
        A RigidTransform representation of the pose's rotation matrix.
    """
    def __init__(self, p, r, x0, face=None, stp_id=-1):
        """Create a new stable pose object.

        Parameters
        ----------
        p : float
            Probability associated with this stable pose.
        r : :obj:`numpy.ndarray` of :obj`numpy.ndarray` of float
            3x3 rotation matrix that rotates the mesh into the stable pose from
            standardized coordinates.
        x0 : :obj:`numpy.ndarray` of float
            3D point in the mesh that is resting on the table.
        face : :obj:`numpy.ndarray`
            3D vector of indices corresponding to vertices forming the resting face
        stp_id : :obj:`str`
            A string identifier for the stable pose
        """
        self.p = p
        self.r = r
        self.x0 = x0
        self.face = face
        self.id = stp_id

        # fix stable pose bug
        if np.abs(np.linalg.det(self.r) + 1) < 0.01:
            self.r[1,:] = -self.r[1,:]

    def __eq__(self, other):
        """ Check equivalence by rotation about the z axis """
        if not isinstance(other, StablePose):
            raise ValueError('Can only compare stable pose objects')
        R0 = self.r
        R1 = other.r
        dR = R1.T.dot(R0)
        theta = 0
        while theta < 2 * np.pi:
            Rz = RigidTransform.z_axis_rotation(theta)
            dR = R1.T.dot(Rz).dot(R0)
            if np.linalg.norm(dR - np.eye(3)) < 1e-5:
                return True
            theta += d_theta
        return False

    @property
    def T_obj_table(self):
        return RigidTransform(rotation=self.r, from_frame='obj', to_frame='table')


    @property
    def T_obj_world(self):
        T_world_obj = RigidTransform(rotation=self.r.T,
                                     translation=self.x0,
                                     from_frame='world',
                                     to_frame='obj')
        return T_world_obj.inverse()
    
