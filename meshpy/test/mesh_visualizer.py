import mayavi.mlab as mv

class MeshVisualizer(object):
    """A class for visualizing meshes.
    """

    def __init__(self, mesh):
        """Initialize a MeshVisualizer.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to apply visualizations to.
        """

        self.mesh_ = mesh

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
        surface = mv.triangular_mesh(self.mesh_.vertices_[:,0],
                                     self.mesh_.vertices_[:,1],
                                     self.mesh_.vertices_[:,2],
                                     self.mesh_.triangles_, representation=style,
                                     color=color, opacity=opacity)
        return surface

