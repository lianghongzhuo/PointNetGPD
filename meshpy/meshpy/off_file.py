"""
File for loading and saving meshes from .OFF files
Author: Jeff Mahler
"""
import os
from . import mesh

class OffFile:
    """
    A .off file reader and writer.

    Attributes
    ----------
    filepath : :obj:`str`
        The full path to the .off file associated with this reader/writer.
    """
    def __init__(self, filepath):
        '''
        Set the path to the file to open
        '''
        self.filepath_ = filepath
        file_root, file_ext = os.path.splitext(self.filepath_)
        if file_ext.lower() != '.off':
            raise Exception('Cannot load file extension %s. Please supply a .off file' %(file_ext))

    @property
    def filepath(self):
        """Returns the full path to the .off file associated with this reader/writer.

        Returns
        -------
        :obj:`str`
            The full path to the .ff file associated with this reader/writer.
        """
        return self.filepath_

    def read(self):
        """Reads in the .off file and returns a Mesh3D representation of that mesh.

        Returns
        -------
        :obj:`Mesh3D`
            A Mesh3D created from the data in the .off file.
        """
        verts = []
        faces = []
        f = open(self.filepath_, 'r')

        # parse header (NOTE: we do not support reading edges)
        header = f.readline()
        tokens = header.split()
        if len(tokens) == 1:
            header = f.readline()
            tokens = header.split()
        else:
            tokens = tokens[1:]
        num_vertices = int(tokens[0])
        num_faces = int(tokens[1])

        # read vertices 
        for i in range(num_vertices):
            line = f.readline()
            tokens = line.split()
            vertex = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
            verts.append(vertex)

        # read faces 
        for i in range(num_faces):
            line = f.readline()
            tokens = line.split()
            if int(tokens[0]) != 3:
                raise ValueError('Only triangle meshes supported, but OFF file has %d-faces' %(int(tokens[0])))
            face = [int(tokens[1]), int(tokens[2]), int(tokens[3])]
            faces.append(face)


        return mesh.Mesh3D(verts, faces)

    def write(self, mesh):
        """Writes a Mesh3D object out to a .off file format

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The Mesh3D object to write to the .obj file.

        Note
        ----
        Does not support material files or texture coordinates.
        """
        raise NotImplementedError()

