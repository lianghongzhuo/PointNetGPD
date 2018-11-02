'''
Reads and writes sdfs to file
Author: Jeff Mahler
'''
import numpy as np
import os

from . import sdf

class SdfFile:
    """
    A Signed Distance Field .sdf file reader and writer.

    Attributes
    ----------
    filepath : :obj:`str`
        The full path to the .sdf or .csv file associated with this reader/writer.
    """
    def __init__(self, filepath):
        """Construct and initialize a .sdf file reader and writer.

        Parameters
        ----------
        filepath : :obj:`str`
            The full path to the desired .sdf or .csv file

        Raises
        ------
        ValueError
            If the file extension is not .sdf of .csv.
        """
        self.filepath_ = filepath
        file_root, file_ext = os.path.splitext(self.filepath_)

        if file_ext == '.sdf':
            self.use_3d_ = True
        elif file_ext == '.csv':
            self.use_3d_ = False
        else:
            raise ValueError('Extension %s invalid for SDFs' %(file_ext))

    @property
    def filepath(self):
        """Returns the full path to the file associated with this reader/writer.

        Returns
        -------
        :obj:`str`
            The full path to the file associated with this reader/writer.
        """
        return self.filepath_

    def read(self):
        """Reads in the SDF file and returns a Sdf object.

        Returns
        -------
        :obj:`Sdf`
            A Sdf created from the data in the file.
        """
        if self.use_3d_:
            return self._read_3d()
        else:
            return self._read_2d()


    def _read_3d(self):
        """Reads in a 3D SDF file and returns a Sdf object.

        Returns
        -------
        :obj:`Sdf3D`
            A 3DSdf created from the data in the file.
        """
        if not os.path.exists(self.filepath_):
            return None

        my_file = open(self.filepath_, 'r')
        nx, ny, nz = [int(i) for i in my_file.readline().split()]     #dimension of each axis should all be equal for LSH
        ox, oy, oz = [float(i) for i in my_file.readline().split()]   #shape origin
        dims = np.array([nx, ny, nz])
        origin = np.array([ox, oy, oz])

        resolution = float(my_file.readline()) # resolution of the grid cells in original mesh coords
        sdf_data = np.zeros(dims)

        # loop through file, getting each value
        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    sdf_data[i][j][k] = float(my_file.readline())
                    count += 1 
        my_file.close()
        return sdf.Sdf3D(sdf_data, origin, resolution)

    def _read_2d(self):
        """Reads in a 2D SDF file and returns a Sdf object.

        Returns
        -------
        :obj:`Sdf2D`
            A 2DSdf created from the data in the file.
        """
        if not os.path.exists(self.filepath_):
            return None

        sdf_data = np.loadtxt(self.filepath_, delimiter=',') 
        return sdf.Sdf2D(sdf_data)

    def write(self, sdf):
        """Writes an SDF to a file.

        Parameters
        ----------
        sdf : :obj:`Sdf`
            An Sdf object to write out.

        Note
        ----
            This is not currently implemented or supported.
        """
        pass

if __name__ == '__main__':
    pass

