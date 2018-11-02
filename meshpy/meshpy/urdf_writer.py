"""
File for loading and saving meshes as URDF files
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np
import os
from subprocess import Popen

import xml.etree.cElementTree as et

from .mesh import Mesh3D
from .obj_file import ObjFile

def split_vhacd_output(mesh_filename):
    """ Splits the output of vhacd into multiple .OBJ files.

    Parameters
    ----------
    mesh_filename : :obj:`str`
        the filename of the mesh from v-hacd

    Returns
    -------
    :obj:`list` of :obj:`str`
        the string filenames of the individual convex pieces
    """
    # read params
    file_root, file_ext = os.path.splitext(mesh_filename)
    f = open(mesh_filename, 'r')
    lines = f.readlines()
    line_num = 0
    num_lines = len(lines)
    num_verts = 0
    vert_offset = 0
    cvx_piece_f = None
    out_filenames = []

    # create a new output .OBJ file for each instance of "{n} convex" in the input file
    while line_num < num_lines:
        line = lines[line_num]
        tokens = line.split()

        # new convex piece
        if tokens[0] == 'o':
            # write old convex piece to file
            if cvx_piece_f is not None:
                cvx_piece_f.close()

            # init new convex piece
            cvx_piece_name = tokens[1]
            out_filename = '%s_%s%s' %(file_root, cvx_piece_name, file_ext)
            logging.info('Writing %s' %(out_filename))
            cvx_piece_f = open(out_filename, 'w')
            vert_offset = num_verts
            out_filenames.append(out_filename)
        # add to vertices
        elif tokens[0] == 'v':
            cvx_piece_f.write(line)
            num_verts += 1
        elif tokens[0] == 'f':
            v0 = int(tokens[1]) - vert_offset
            v1 = int(tokens[2]) - vert_offset
            v2 = int(tokens[3]) - vert_offset
            f_line = 'f %d %d %d\n' %(v0, v1, v2)
            cvx_piece_f.write(f_line)

        line_num += 1

    # close the file
    if cvx_piece_f is not None:
        cvx_piece_f.close()
    return out_filenames

def convex_decomposition(mesh, cache_dir='', name='mesh'):
    """ Performs a convex deomposition of the mesh using V-HACD.
    
    Parameters
    ----------
    cache_dir : str
        a directory to store the intermediate files
    name : str
        the name of the mesh for the cache file

    Returns
    -------
    :obj:`list` of :obj:`Mesh3D`
        list of mesh objects comprising the convex pieces of the object, or None if vhacd failed
    :obj:`list` of str
        string file roots of the convex pieces
    float
        total volume of the convex pieces
    """
    # save to file
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    obj_filename = os.path.join(cache_dir, '%s.obj' %(name))
    vhacd_out_filename = os.path.join(cache_dir, '%s_vhacd.obj' %(name))
    log_filename = os.path.join(cache_dir, 'vhacd_log.txt')
    print(obj_filename)
    ObjFile(obj_filename).write(mesh)

    # use v-hacd for convex decomposition
    cvx_decomp_cmd = 'vhacd --input %s --output %s --log %s' %(obj_filename,
                                                               vhacd_out_filename,
                                                               log_filename)
    vhacd_process = Popen(cvx_decomp_cmd, bufsize=-1, close_fds=True, shell=True)
    vhacd_process.wait()

    # check success
    if not os.path.exists(vhacd_out_filename):
        logging.error('Output mesh file %s not found. V-HACD failed. Is V-HACD installed?' %(vhacd_out_filename))
        return None

    # create separate convex piece files
    convex_piece_files = split_vhacd_output(vhacd_out_filename)

    # read convex pieces
    convex_piece_meshes = []
    convex_piece_filenames = []
    convex_pieces_volume = 0.0

    # read in initial meshes for global properties
    for convex_piece_filename in convex_piece_files:

        # read in meshes
        obj_file_path, obj_file_root = os.path.split(convex_piece_filename)
        of = ObjFile(convex_piece_filename)
        convex_piece = of.read()
        convex_pieces_volume += convex_piece.total_volume()
        convex_piece_meshes.append(of.read())
        convex_piece_filenames.append(obj_file_root)

    return convex_piece_meshes, convex_piece_filenames, convex_pieces_volume

class UrdfWriter(object):
    """
    A .urdf file writer.

    Attributes
    ----------
    filepath : :obj:`str`
        The full path to the .urdf file associated with this writer.
    """

    def __init__(self, filepath):
        """Construct and initialize a .urdf file writer.

        Parameters
        ----------
        filepath : :obj:`str`
            The full path to the directory in which to save the URDF file

        Raises
        ------
        ValueError
            If the fullpath is not a directory
        """
        self.filepath_ = filepath
        file_root, file_ext = os.path.splitext(self.filepath_)
        file_path, file_name = os.path.split(file_root)
        self.name_ = file_name
        if file_ext != '':
            raise ValueError('URDF path must be a directory')

    @property
    def filepath(self):
        """Returns the full path to the URDF directory associated with this writer.

        Returns
        -------
        :obj:`str`
            The full path to the URDF directory associated with this writer.
        """
        return self.filepath_

    @property
    def urdf_filename(self):
        """Returns the full path to the URDF file associated with this writer.

        Returns
        -------
        :obj:`str`
            The full path to the URDF file associated with this writer.
        """
        return os.path.join(self.filepath_, '%s.urdf' %(self.name_))

    def write(self, mesh):
        """Writes a Mesh3D object to a .urdf file.
        First decomposes the mesh using V-HACD, then writes to a .URDF

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The Mesh3D object to write to the .urdf file.

        Note
        ----
        Requires v-hacd installation.
        Does not support moveable joints.
        """
        # perform convex decomp
        convex_piece_meshes, convex_piece_filenames, convex_pieces_volume = convex_decomposition(mesh, cache_dir=self.filepath_, name=self.name_)

        # get the masses and moments of inertia
        effective_density = mesh.total_volume() / convex_pieces_volume

        # open an XML tree
        root = et.Element('robot', name='root')

        # loop through all pieces
        prev_piece_name = None
        for convex_piece, filename in zip(convex_piece_meshes, convex_piece_filenames):
            # set the mass properties
            convex_piece.center_of_mass = mesh.center_of_mass
            convex_piece.density = effective_density * mesh.density
            
            _, file_root = os.path.split(filename)
            file_root, _ = os.path.splitext(file_root)
            obj_filename = 'package://%s/%s' %(self.name_, filename)

            # write to xml
            piece_name = 'link_%s'%(file_root)
            I = convex_piece.inertia
            link = et.SubElement(root, 'link', name=piece_name)
            
            inertial = et.SubElement(link, 'inertial')
            origin = et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
            mass = et.SubElement(inertial, 'mass', value='%.2E'%convex_piece.mass)
            inertia = et.SubElement(inertial, 'inertia', ixx='%.2E'%I[0,0], ixy='%.2E'%I[0,1], ixz='%.2E'%I[0,2],
                                    iyy='%.2E'%I[1,1], iyz='%.2E'%I[1,2], izz='%.2E'%I[2,2])
            
            visual = et.SubElement(link, 'visual')
            origin = et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
            geometry = et.SubElement(visual, 'geometry')
            mesh_element = et.SubElement(geometry, 'mesh', filename=obj_filename)
            material = et.SubElement(visual, 'material', name='')
            color = et.SubElement(material, 'color', rgba="0.75 0.75 0.75 1")
            
            collision = et.SubElement(link, 'collision')
            origin = et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")            
            geometry = et.SubElement(collision, 'geometry')
            mesh_element = et.SubElement(geometry, 'mesh', filename=obj_filename)
            
            if prev_piece_name is not None:
                joint = et.SubElement(root, 'joint', name='%s_joint'%(piece_name), type='fixed')
                origin = et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
                parent = et.SubElement(joint, 'parent', link=prev_piece_name)
                child = et.SubElement(joint, 'child', link=piece_name)
                
            prev_piece_name = piece_name
                
        # write URDF file
        tree = et.ElementTree(root)
        tree.write(self.urdf_filename)
    
        # write config file
        root = et.Element('model')
        model = et.SubElement(root, 'name')
        model.text = self.name_
        version = et.SubElement(root, 'version')
        version.text = '1.0'
        sdf = et.SubElement(root, 'sdf', version='1.4')
        urdf_root, urdf_ext = os.path.splitext(self.urdf_filename)
        urdf_path, urdf_name = os.path.split(urdf_root)
        sdf.text = urdf_name
        
        author = et.SubElement(root, 'author')    
        et.SubElement(author, 'name').text = 'AUTOLAB meshpy'
        et.SubElement(author, 'email').text = 'jmahler@berkeley.edu'
        
        description = et.SubElement(root, 'description')        
        description.text = 'My awesome %s' %(self.name_)
        
        tree = et.ElementTree(root)
        config_filename = os.path.join(self.filepath_, 'model.config')
        tree.write(config_filename)

    def write_pieces(self, meshes, center_of_mass=np.zeros(3), density=1.0):
        """Writes a list of Mesh3D object to a .urdf file.

        Parameters
        ----------
        meshes : :obj:`list` of :obj:`Mesh3D`
            The Mesh3D objects to write to the .urdf file.
        center_of_mass : :obj:`numpy.ndarray`
            The center of mass of the combined object. Defaults to zero.
        desnity : float
            The density fo the mesh pieces

        Note
        ----
        Does not support moveable joints.
        """
        # create output directory
        out_dir = self.filepath_
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # read convex pieces
        mesh_filenames = []

        # write meshes to reference with URDF files
        for i, mesh in enumerate(meshes):
            # read in meshes
            obj_file_root = '%s_%04d.obj' %(self.name_, i)
            obj_filename = os.path.join(out_dir, obj_file_root)
            ObjFile(obj_filename).write(mesh)
            mesh_filenames.append(obj_file_root)

        # open an XML tree
        root = et.Element('robot', name='root')

        # loop through all pieces
        prev_piece_name = None
        for mesh, filename in zip(meshes, mesh_filenames):
            # set the mass properties
            mesh.center_of_mass = center_of_mass
            mesh.density = density
            
            _, file_root = os.path.split(filename)
            file_root, _ = os.path.splitext(file_root)
            obj_filename = 'package://%s/%s' %(self.name_, filename)

            # write to xml
            piece_name = 'link_%s'%(file_root)
            I = mesh.inertia
            link = et.SubElement(root, 'link', name=piece_name)
            
            inertial = et.SubElement(link, 'inertial')
            origin = et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
            mass = et.SubElement(inertial, 'mass', value='%.2E'%mesh.mass)
            inertia = et.SubElement(inertial, 'inertia', ixx='%.2E'%I[0,0], ixy='%.2E'%I[0,1], ixz='%.2E'%I[0,2],
                                    iyy='%.2E'%I[1,1], iyz='%.2E'%I[1,2], izz='%.2E'%I[2,2])
            
            visual = et.SubElement(link, 'visual')
            origin = et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
            geometry = et.SubElement(visual, 'geometry')
            mesh_element = et.SubElement(geometry, 'mesh', filename=obj_filename)
            material = et.SubElement(visual, 'material', name='')
            color = et.SubElement(material, 'color', rgba="0.75 0.75 0.75 1")
            
            collision = et.SubElement(link, 'collision')
            origin = et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")            
            geometry = et.SubElement(collision, 'geometry')
            mesh_element = et.SubElement(geometry, 'mesh', filename=obj_filename)
            
            if prev_piece_name is not None:
                joint = et.SubElement(root, 'joint', name='%s_joint'%(piece_name), type='fixed')
                origin = et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
                parent = et.SubElement(joint, 'parent', link=prev_piece_name)
                child = et.SubElement(joint, 'child', link=piece_name)
                
            prev_piece_name = piece_name
                
        # write URDF file
        tree = et.ElementTree(root)
        tree.write(self.urdf_filename)
    
        # write config file
        root = et.Element('model')
        model = et.SubElement(root, 'name')
        model.text = self.name_
        version = et.SubElement(root, 'version')
        version.text = '1.0'
        sdf = et.SubElement(root, 'sdf', version='1.4')
        urdf_root, urdf_ext = os.path.splitext(self.urdf_filename)
        urdf_path, urdf_name = os.path.split(urdf_root)
        sdf.text = urdf_name
        
        author = et.SubElement(root, 'author')    
        et.SubElement(author, 'name').text = 'AUTOLAB meshpy'
        et.SubElement(author, 'email').text = 'jmahler@berkeley.edu'
        
        description = et.SubElement(root, 'description')        
        description.text = 'My awesome %s' %(self.name_)
        
        tree = et.ElementTree(root)
        config_filename = os.path.join(out_dir, 'model.config')
        tree.write(config_filename)


