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
Wrappers for DexNet databases
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import datetime as dt
import gc
import h5py
import logging
import numbers
import numpy as np
import os
from subprocess import Popen
import sys
import time

from dexnet.constants import *

from dexnet.database import Hdf5ObjectFactory
from dexnet.database.keys import *

from dexnet.grasping import GraspableObject3D

import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file
import meshpy.stp_file as stp_file
from meshpy import UrdfWriter

from perception import RenderMode

import IPython

try:
    import dill
except ImportError:
    logging.warning("Could not import dill, some metadata operations will be unavailable")

INDEX_FILE = 'index.db'


# class Database(object, metaclass=ABCMeta):
class Database(object):
    """ Abstract class for Dex-Net databases. Main purpose is to wrap individual datasets.

    Attributes
    ----------
    access_level : :obj:`str`
        level of access, READ_ONLY versus READ_WRITE versus WRITE, for the database
    """
    __metaclass__ = ABCMeta

    def __init__(self, access_level=READ_ONLY_ACCESS):
        self.access_level_ = access_level

    @property
    def access_level(self):
        return self.access_level_


class Hdf5Database(Database):
    """ Wrapper for HDF5 databases with h5py.

    Attributes
    ----------
    database_filename : :obj:`str`
        filenaame of HDF5 database, must have extension .hdf5
    access_level : :obj:`str`
        level of access, READ_ONLY versus READ_WRITE versus WRITE, for the database
    cache_dir : :obj:`str`
        directory to cache files used by the database wrapper class
    datasets : :obj:`list` of :obj:`Hdf5Dataset`
        datasets contained in this database
    """
    def __init__(self, database_filename, access_level=READ_ONLY_ACCESS,
                 cache_dir='.dexnet'):
        Database.__init__(self, access_level)
        self.database_filename_ = database_filename
        if not self.database_filename_.endswith(HDF5_EXT):
            raise ValueError('Must provide HDF5 database')

        self.database_cache_dir_ = cache_dir
        if not os.path.exists(self.database_cache_dir_):
            os.mkdir(self.database_cache_dir_)

        self._load_database()
        self._load_datasets()

    def _create_new_db(self):
        """ Creates a new database """
        self.data_ = h5py.File(self.database_filename_, 'w')

        dt_now = dt.datetime.now()
        creation_stamp = '%s-%s-%s-%sh-%sm-%ss' %(dt_now.month, dt_now.day, dt_now.year, dt_now.hour, dt_now.minute, dt_now.second) 
        self.data_.attrs[CREATION_KEY] = creation_stamp
        self.data_.create_group(DATASETS_KEY)

    def _load_database(self):
        """ Loads in the HDF5 file """
        if self.access_level == READ_ONLY_ACCESS:
            self.data_ = h5py.File(self.database_filename_, 'r')
        elif self.access_level == READ_WRITE_ACCESS:
            if os.path.exists(self.database_filename_):
                self.data_ = h5py.File(self.database_filename_, 'r+')
            else:
                self._create_new_db()
        elif self.access_level == WRITE_ACCESS:
            self._create_new_db()
        self.dataset_names_ = list(self.data_[DATASETS_KEY].keys())

    def _load_datasets(self):
        """ Load in the datasets """
        self.datasets_ = []
        for dataset_name in self.dataset_names_:
            if dataset_name not in list(self.data_[DATASETS_KEY].keys()):
                logging.warning('Dataset %s not in database' %(dataset_name))
            else:
                dataset_cache_dir = os.path.join(self.database_cache_dir_, dataset_name)
                self.datasets_.append(Hdf5Dataset(dataset_name, self.data_[DATASETS_KEY][dataset_name],
                                                  cache_dir=dataset_cache_dir))

    @property
    def cache_dir(self):
        return self.database_cache_dir_

    @property
    def datasets(self):
        return self.datasets_

    def dataset(self, dataset_name):
        """ Returns handles to individual Hdf5 datasets.

        Parameters
        ----------
        dataset_name : :obj:`str`
            string name of the dataset

        Returns
        -------
        :obj `Hdf5Dataset`
            dataset wrapper for the given name, if it exists, None otherwise
        """
        if self.datasets is None or dataset_name not in self.dataset_names_:
            return None
        for dataset in self.datasets_:
            if dataset.name == dataset_name:
                return dataset

    def flush(self):
        """ Flushes the file """
        self.data_.flush()
        gc.collect()

    def close(self):
        """ Close the HDF5 file """
        self.data_.close()

    def __getitem__(self, dataset_name):
        """ Dataset name indexing using bracketing """
        return self.dataset(dataset_name)
        
    # New dataset creation / modification functions
    def create_dataset(self, dataset_name, obj_keys=[]):
        """ Create dataset with obj keys and add to the database.

        Parameters
        ----------
        dataset_name : :obj:`str`
            name of dataset to create
        obj_keys : :obj:`list` of :obj:`str`
            keys of the objects to add to the dataset

        Returns
        -------
        :obj:`Hdf5Dataset`
            the created dataset
        """
        if dataset_name in list(self.data_[DATASETS_KEY].keys()):
            logging.warning('Dataset %s already exists. Cannot overwrite' %(dataset_name))
            return self.datasets_[list(self.data_[DATASETS_KEY].keys()).index(dataset_name)]
        self.data_[DATASETS_KEY].create_group(dataset_name)
        self.data_[DATASETS_KEY][dataset_name].create_group(OBJECTS_KEY)
        self.data_[DATASETS_KEY][dataset_name].create_group(METRICS_KEY)
        for obj_key in obj_keys:
            self.data_[DATASETS_KEY][dataset_name][OBJECTS_KEY].create_group(obj_key)

        dataset_cache_dir = os.path.join(self.database_cache_dir_, dataset_name)
        self.dataset_names_.append(dataset_name)
        self.datasets_.append(Hdf5Dataset(dataset_name, self.data_[DATASETS_KEY][dataset_name],
                                          cache_dir=dataset_cache_dir))
        return self.datasets_[-1] # return the dataset
        
    def create_linked_dataset(self, dataset_name, graspable_list, nearest_neighbors):
        """ Creates a new dataset that links to objects physically stored as part of another dataset. Not currently implemented """
        raise NotImplementedError()

class Dataset(object):
    """ Abstract class for different dataset implementations. Only Hdf5 is currently supported """
    pass

class Hdf5Dataset(Dataset):
    """ Wrapper for HDF5 datasets with h5py.

    Attributes
    ----------
    dataset_name : :obj:`str`
        fname of the dataset
    data : :obj:`h5py.Group`
        Hdf5 group corresponding to the data
    cache_dir : :obj:`str`
        directory to cache files used by the dataset
    start_index : :obj:`int`
        initial object index to use for iteration
    end_index : :obj:`int`
        final object index to use for iteration
    """
    def __init__(self, dataset_name, data, cache_dir=None,
                 start_index=0, end_index=None):
        self.dataset_name_ = dataset_name
        self.data_ = data
        self.object_keys_ = None
        self.start_index_ = start_index
        self.end_index_ = end_index
        if self.end_index_ is None:
            self.end_index_ = len(list(self.objects.keys()))

        self.cache_dir_ = cache_dir
        if self.cache_dir_ is None:
            self.cache_dir_ = os.path.join('.dexnet', self.dataset_name_)
        if not os.path.exists(self.cache_dir_):
            os.mkdir(self.cache_dir_)
            
        self._metadata_functions = {}

    @property
    def name(self):
        """ :obj:`str` : Name of the dataset """
        return self.dataset_name_

    @property
    def objects(self):
        """ :obj:`h5py.Group` : Data containing handles of objects.
        Acts like a dictionary mapping object keys to object data.
        """
        return self.data_[OBJECTS_KEY]

    @property
    def object_keys(self):
        """ :obj:`list` of :obj:`str` : Names of all objects in the dataset.
        """
        if not self.object_keys_:
            self.object_keys_ = list(self.objects.keys())[self.start_index_:self.end_index_]
        return self.object_keys_

    @property
    def num_objects(self):
        """ int : number of objects in the dataset. """
        return len(self.object_keys)

    # easy data accessors
    @property
    def metrics(self):
        """ :obj:`h5py.Group` : Data for available metrics in the dataset
        """
        if METRICS_KEY in list(self.data_.keys()):
            return self.data_[METRICS_KEY]
        return None

    @property
    def metric_names(self):
        """ :obj:`list` of :obj:`str` : Names of aviailable metrics
        """
        if self.metrics is None: return []
        return list(self.metrics.keys())
    
    @property
    def metadata(self):
        """ :obj:`h5py.Group` : Types and optional descriptions/functions for metadata in the dataset
        """
        if METADATA_KEY in list(self.data_.keys()):
            return self.data_[METADATA_KEY]
        return None
    
    @property
    def metadata_names(self):
        """:obj:`list` of :obj:`str` : Names of available metadata
        """
        if self.metadata is None: return [] 
        return list(self.metadata.keys())

    def object(self, key):
        """ :obj:`h5py.Group` : Data for objects in the dataset """
        return self.objects[key]

    def sdf_data(self, key):
        return self.objects[key][SDF_KEY]

    def mesh_data(self, key):
        return self.objects[key][MESH_KEY]

    def convex_piece_data(self, key):
        if CONVEX_PIECES_KEY in list(self.objects[key].keys()):
            return self.objects[key][CONVEX_PIECES_KEY]
        return None

    def grasp_data(self, key, gripper=None):
        if gripper:
            return self.objects[key][GRASPS_KEY][gripper]
        return self.objects[key][GRASPS_KEY]

    def stable_pose_data(self, key, stable_pose_id=None):
        if stable_pose_id is not None:
            self.objects[key][STP_KEY][stable_pose_id]
        return self.objects[key][STP_KEY]

    def category(self, key):
        return self.objects[key].attrs[CATEGORY_KEY]

    def rendered_image_data(self, key, stable_pose_id=None, image_type=None):
        if stable_pose_id is not None and image_type is not None:
            return self.stable_pose_data(key)[stable_pose_id][RENDERED_IMAGES_KEY][image_type]
        elif stable_pose_id is not None:
            return self.stable_pose_data(key)[stable_pose_id][RENDERED_IMAGES_KEY]
        elif image_type is not None:
            if image_type not in RENDERED_IMAGE_TYPES:
                raise ValueError('Image type %s not supported' %(image_type))
            return self.object(key)[RENDERED_IMAGES_KEY][image_type]
        return self.object(key)[RENDERED_IMAGES_KEY]

    def metric_data(self, metric):
        if metric in list(self.metrics.keys()):
            return self.metrics[metric]
        return None
    
    def metadata_data(self, metadata):
        if metadata in list(self.metadata.keys()):
            return self.metadata[metadata]
        return None

    # iterators
    def __getitem__(self, index):
        """ Index a particular object in the dataset """
        if isinstance(index, numbers.Number):
            if index < 0 or index >= len(self.object_keys):
                raise ValueError('Index out of bounds. Dataset contains %d objects' %(len(self.object_keys)))
            obj = self.graspable(self.object_keys[index])
            return obj
        elif isinstance(index, str):
            obj = self.graspable(index)
            return obj

    def __iter__(self):
        """ Generate iterator """
        self.iter_count_ = self.start_index_ # NOT THREAD SAFE!
        return self

    def subset(self, start_index, end_index):
        """ Returns a subset of the dataset (should be used for iterating only)
        
        Parameters
        ----------
        start_index : :obj:`int`
            index of first object key in subset
        end_index : :obj:`int`
            index of last object key in subset

        Returns
        -------
        :obj:`Hdf5Dataset`
            Dataset containing only the specified subset
        """
        return Hdf5Dataset(self.dataset_name_, self.data_, self.cache_dir_,
                           start_index, end_index) 
    
    def __next__(self):
        """ Read the next object file in the list.
        
        Returns
        -------
        :obj:`GraspableObject3D`
            the next graspable object in the iteration
        """
        if self.iter_count_ >= len(self.object_keys) or self.iter_count_ >= self.end_index_:
            raise StopIteration
        else:
            logging.info('Returning datum %s' %(self.object_keys[self.iter_count_]))
            try:
                obj = self.graspable(self.object_keys[self.iter_count_])    
            except:
                logging.warning('Error reading %s. Skipping' %(self.object_keys[self.iter_count_]))
                self.iter_count_ = self.iter_count_ + 1
                return next(self)

            self.iter_count_ = self.iter_count_ + 1
            return obj

    # direct reading / writing
    def graspable(self, key):
        """Read in the GraspableObject3D corresponding to given key.

        Parameters
        ---------
        key : :obj:`str`
            the string key of the object to index

        Returns
        -------
        :obj:`GraspableObject3D`
            the graspable object corresponding to the given key

        Raises
        ------
        ValueError
            If the key is not found in the dataset
        """
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))

        # read in data
        sdf = Hdf5ObjectFactory.sdf_3d(self.sdf_data(key))
        mesh = Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))
        mass = self.object(key).attrs[MASS_KEY]
        convex_pieces = None
        if self.convex_piece_data(key) is not None:
            convex_pieces = []
            for piece_key in list(self.convex_piece_data(key).keys()):
                convex_pieces.append(Hdf5ObjectFactory.mesh_3d(self.convex_piece_data(key)[piece_key]))
        return GraspableObject3D(sdf, mesh=mesh, key=key,
                                 model_name=self.obj_mesh_filename(key),
                                 mass=mass, convex_pieces=convex_pieces)

    def create_graspable(self, key, mesh=None, sdf=None, stable_poses=None, mass=1.0):
        """ Creates a graspable object in the given dataset

        Parameters
        ----------
        key : :obj:`str`
            name of the object, for indexing
        mesh : :obj:`Mesh3D`
            mesh for object geometry
        sdf : :obj:`Sdf3D`
            signed distance field for object contact computations
        stable_poses : :obj:`list` of :obj:`StablePose`
            stable poses of the object
        mass : float
            mass of the object in kilograms

        Raises
        -------
        ValueError
            If the key is already in the dataset
        """
        if key in self.object_keys:
            raise ValueError('Object %s already exists!' %(key))

        # create object tree
        self.objects.create_group(key)
        self.object(key).create_group(MESH_KEY)
        self.object(key).create_group(SDF_KEY)
        self.object(key).create_group(STP_KEY)
        self.object(key).create_group(RENDERED_IMAGES_KEY)
        self.object(key).create_group(SENSOR_DATA_KEY)
        self.object(key).create_group(GRASPS_KEY)

        # add the different pieces if provided
        if sdf:
            Hdf5ObjectFactory.write_sdf_3d(sdf, self.sdf_data(key))
        if mesh:
            Hdf5ObjectFactory.write_mesh_3d(mesh, self.mesh_data(key))
        if stable_poses:
            Hdf5ObjectFactory.write_stable_poses(stable_poses, self.stable_pose_data(key))

        # add the attributes
        self.object(key).attrs.create(MASS_KEY, mass)

        # force re-read of keys
        self.object_keys_ = None
        self.end_index_ = len(list(self.objects.keys()))

    def store_mesh(self, key, mesh, force_overwrite=False):
        """ Associates a mesh with the given object.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        mesh : :obj:`meshpy.Mesh3D`
            mesh to store
        force_overwrite : bool
            whether or not to overwrite

        Returns
        -------
        bool
            True if mesh stored for the given object, False otherwise
        """
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))
        if self.mesh_data(key) is not None and not force_overwrite:
            raise ValueError('Mesh for key %s already exist specified and force overwrite not specified' %(key))

        # delete old mesh data
        del self.object(key)[MESH_KEY]

        # write mesh
        self.object(key).create_group(MESH_KEY)        
        Hdf5ObjectFactory.write_mesh_3d(mesh, self.mesh_data(key))
        return True

    def store_convex_pieces(self, key, convex_pieces, force_overwrite=False):
        """ Associates convex pieces with the given object.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        convex_pieces : :obj:`list` of :obj:`meshpy.Mesh3D`
            convex pieces to store
        force_overwrite : bool
            whether or not to overwrite convex pieces

        Returns
        -------
        bool
            True if convex pieces were stored for the given object, False otherwise
        """
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))
        if self.convex_piece_data(key) is not None and not force_overwrite:
            raise ValueError('Convex pieces for key %s already exist specified and force overwrite not specified' %(key))

        # create convex piece data if necessary
        if self.convex_piece_data(key) is None:
            self.object(key).create_group(CONVEX_PIECES_KEY)
        
        # add pieces one by one
        for i, convex_piece in enumerate(convex_pieces):
            piece_key = 'piece_%03d' %(i)
            self.convex_piece_data(key).create_group(piece_key)
            Hdf5ObjectFactory.write_mesh_3d(convex_piece, self.convex_piece_data(key)[piece_key])
        return True
           
    def store_stable_poses(self, key, stable_poses, force_overwrite=False):
        """ Associates stable poses with the given object.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            stable poses to store
        force_overwrite : bool
            whether or not to overwrite stable poses

        Returns
        -------
        bool
            True if grasps were stored for the given object, False otherwise
        """
        Hdf5ObjectFactory.write_stable_poses(stable_poses, self.stable_pose_data(key), force_overwrite=force_overwrite)
        return True

    def delete_graspable(self, key):
        """ Delete a graspable from the dataset.
        
        Parameters
        ----------
        key : :obj:`str`
            name of the object to delete            

        Returns
        -------
        bool
            True if object deleted, False otherwise
        """
        if key not in list(self.objects.keys()):
            logging.warning('Graspable %s not found. Nothing to delete' %(key))
            return False

        del self.object(key)[MESH_KEY]
        del self.object(key)[SDF_KEY]
        del self.object(key)[STP_KEY]
        del self.object(key)[RENDERED_IMAGES_KEY]
        del self.object(key)[SENSOR_DATA_KEY]
        del self.object(key)[GRASPS_KEY]
        del self.objects[key]

        # force re-read of keys
        self.object_keys_ = None
        self.end_index_ = len(list(self.objects.keys()))

        return True

    def delete_convex_pieces(self, key):
        """ Delete convex pieces for an object from the dataset.
        
        Parameters
        ----------
        key : :obj:`str`
            name of the object to delete pieces for

        Returns
        -------
        bool
            True if object deleted, False otherwise
        """
        if key not in list(self.objects.keys()):
            logging.warning('Graspable %s not found. Nothing to delete' %(key))
            return False
        if self.convex_piece_data(key) is not None:
            del self.object(key)[CONVEX_PIECES_KEY]
        return True

    def obj_mesh_filename(self, key, scale=1.0, output_dir=None, overwrite=False):
        """ Writes an obj file in the database "cache"  directory and returns the path to the file.
        Does not overwrite existing files by default.
        Typically used for integration with other libraries that require mesh files as .obj.

        Parameters
        ---------
        key : :obj:`str`
            key of object to write mesh for
        scale : float
            optional rescaling factor
        output_dir : :obj:`str`
            directory to save to, if None saves to cache dir
        overwrite : bool
            whether or not to overwrite an existing file with the same name

        Returns
        -------
        :obj:`str`
            filename of .obj file
        """
        # read mesh
        mesh = Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))
        mesh.rescale(scale)
        if output_dir is None:
            output_dir = self.cache_dir_
        obj_filename = os.path.join(output_dir, key + OBJ_EXT)

        # write mesh as an obj file if necessary
        if overwrite or not os.path.exists(obj_filename):
            of = obj_file.ObjFile(obj_filename)
            of.write(mesh)
        return obj_filename

    def stl_mesh_filename(self, key, scale=1.0, output_dir=None, overwrite=False):
        """ Writes an stl file in the database "cache"  directory and returns the path to the file.
        Does not overwrite existing files by default.
        Typically used for integration with other libraries that require mesh files as .obj.

        Parameters
        ---------
        key : :obj:`str`
            key of object to write mesh for
        scale : float
            optional rescaling factor
        output_dir : :obj:`str`
            directory to save to, if None saves to cache dir
        overwrite : bool
            whether or not to overwrite an existing file with the same name

        Returns
        -------
        :obj:`str`
            filename of .stl file
        """
        # write tmp obj file
        obj_filename = self.obj_mesh_filename(key, scale=scale, output_dir=output_dir)
        if output_dir is None:
            output_dir = self.cache_dir_

        # convert to stl
        stl_filename = os.path.join(output_dir, key + STL_EXT)
        if overwrite or not os.path.exists(stl_filename):
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(obj_filename, stl_filename)
            os.system(meshlabserver_cmd)
        return stl_filename

    def urdf_mesh_filename(self, key, output_dir=None, overwrite=True):
        """ Writes a urd file in the database "cache"  directory and returns the path to the file.
        Overwrites existing files by default.
        Typically used for integration with other libraries that require mesh files as .obj.

        Parameters
        ---------
        key : :obj:`str`
            key of object to write mesh for
        scale : float
            optional rescaling factor
        output_dir : :obj:`str`
            directory to save to, if None saves to cache dir
        overwrite : bool
            whether or not to overwrite an existing file with the same name

        Returns
        -------
        :obj:`str`
            filename of .urdf file
        """
        # create output dir 
        if output_dir is None:
            output_dir = self.cache_dir_
        urdf_dir = os.path.join(output_dir, key)
        if not os.path.exists(urdf_dir):
            os.mkdir(urdf_dir)

        # read mesh
        mesh = Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))

        # write file
        writer = UrdfWriter(urdf_dir)
        writer.write(mesh)
        return writer.urdf_filename

    # mesh data
    def mesh(self, key):
        """ Read the mesh for the given key.

        Parameters
        ----------
        key : :obj:`str`
            key of object to read mesh for

        Returns
        -------
        :obj:`Mesh3D`
            mesh of object

        Raises
        ------
        ValueError : If the key is not in the dataset
        """
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))
        return Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))

    def convex_pieces(self, key):
        """ Read the set of convex pieces for the given key, if they exist.

        Parameters
        ----------
        key : :obj:`str`
            key of object to read convex pieces for

        Returns
        -------
        :obj:`list` of :obj:`Mesh3D`
            list of convex pieces of object, or None if they do not exist

        Raises
        ------
        ValueError : If the key is not in the dataset
        """
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))
        if self.convex_piece_data(key) is None:
            return None
        
        # read convex pieces
        convex_pieces = []
        for convex_piece_key in list(self.convex_piece_data(key).keys()):
            convex_pieces.append(Hdf5ObjectFactory.mesh_3d(self.convex_piece_data(key)[convex_piece_key]))
        return convex_pieces

    # metric data
    def create_metric(self, metric_name, metric_config):
        """ Creates a grasp quality metric with the given name for easier access.

        Parameters
        ----------
        metric_name : :obj:`str`
            name of new grasp quality metric
        metric_config : :obj:`GraspQualityConfig`
            configuration specifying parameters of a given metric

        Returns
        -------
        bool
            True if creation was successful, False otherwise
        """
        # create metric data if nonexistent
        if self.metrics is None:
            self.data_.create_group(METRICS_KEY)
                        
        if metric_name in list(self.metrics.keys()):
            logging.warning('Metric %s already exists. Aborting...' %(metric_name))
            return False
        self.metrics.create_group(metric_name)

        # add configuration
        metric_group = self.metric_data(metric_name)
        for key in list(metric_config.keys()):
            if isinstance(metric_config[key], dict):
                metric_group.create_group(key)
                for k, v in metric_config[key].items():
                    metric_group[key].attrs.create(k, metric_config[key][k])
            else:
                metric_group.attrs.create(key, metric_config[key])
        return True

    def metric(self, metric_name):
        """ Reads a metric config from the database.

        Parameters
        ----------
        metric_name : :obj:`str`
            name of the metric to read
 
        Returns
        -------
        :obj:`GraspQualityConfig`
            configuration of grasp metric, None if metric does not exist
        """
        # create metric data if nonexistent
        if self.metrics is None:
            logging.warning('No metrics available')
            return None
        if metric_name not in list(self.metrics.keys()):
            logging.warning('Metric %s does not exist. Aborting...' %(metric_name))
            return None

        # read configuration
        metric_group = self.metric_data(metric_name)
        metric_config = {}
        for key in list(metric_group.keys()):
            metric_config[key] = {}
            for k, v in metric_group[key].attrs.items():
                metric_config[key][k] = v
        for key, value in metric_group.attrs.items():
            metric_config[key] = value
        return metric_config

    def has_metric(self, metric_name):
        """ Checks if a metric already exists """
        if metric_name in self.metric_names:
            return True
        return False

    def delete_metric(self, metric_name):
        """ Deletes a metric from the database.

        Parameters
        ----------
        metric_name : :obj:`str`
            name of metric to delete
        """
        if metric_name in self.metric_names:
            del self.metrics[metric_name]

    def available_metrics(self, key, gripper='pr2'):
        """ Returns a list of the metrics computed for a given object.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        gripper : :obj:`str`
            name of gripper

        Returns
        -------
        :obj:`list` of :obj:`str`
            list of names of metric computed and stored for the object
        """
        grasps = self.grasps(key, gripper=gripper)
        gm = self.grasp_metrics(key, grasps, gripper=gripper)
        metrics = set()
        for grasp in grasps:
            metrics.update(list(gm[grasp.id].keys()))
        return list(metrics)

    # grasp data
    def grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Returns the list of grasps for the given graspable and gripper, optionally associated with the given stable pose.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose

        Returns
        -------
        :obj:`list` of :obj:`dexnet.grasping.ParallelJawPtGrasp3D`
            stored grasps for the object and gripper, empty list if gripper not found
        """
        if gripper not in list(self.grasp_data(key).keys()):
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return []
        return Hdf5ObjectFactory.grasps(self.grasp_data(key, gripper))

    def sorted_grasps(self, key, metric, gripper='pr2', stable_pose_id=None):
        """ Returns the list of grasps for the given graspable sorted by decreasing quality according to the given metric.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        metric : :obj:`str`
            name of metric to use for sorting
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose

        Returns
        -------
        :obj:`list` of :obj:`dexnet.grasping.ParallelJawPtGrasp3D`
            stored grasps for the object and gripper sorted by metric in descending order, empty list if gripper not found
        :obj:`list` of float
            values of metrics for the grasps sorted in descending order, empty list if gripper not found
        """
        grasps = self.grasps(key, gripper=gripper, stable_pose_id=stable_pose_id)
        if len(grasps) == 0:
            return [], []
        
        grasp_metrics = self.grasp_metrics(key, grasps, gripper=gripper, stable_pose_id=stable_pose_id)
        if metric not in list(grasp_metrics[list(grasp_metrics.keys())[0]].keys()):
            raise ValueError('Metric %s not recognized' %(metric))

        grasps_and_metrics = [(g, grasp_metrics[g.id][metric]) for g in grasps]
        grasps_and_metrics.sort(key=lambda x: x[1], reverse=True)
        sorted_grasps = [g[0] for g in grasps_and_metrics]
        sorted_metrics = [g[1] for g in grasps_and_metrics]
        return sorted_grasps, sorted_metrics

    def has_grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Checks if grasps exist for a given object.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose

        Returns
        -------
        bool
            True if dataset contains grasps for the given object and gripper, False otherwise
        """
        if gripper not in list(self.grasp_data(key).keys()):
            return False
        return True

    def delete_grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Deletes a set of grasps associated with the given gripper.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose

        Returns
        -------
        bool
            True if grasps were deleted for the given object and gripper, False otherwise
        """
        if gripper not in list(self.grasp_data(key).keys()):
            logging.warning('Gripper type %s not found. Nothing to delete' %(gripper))
            return False
        del self.grasp_data(key)[gripper]
        return True

    def store_grasps(self, key, grasps, gripper='pr2', stable_pose_id=None, force_overwrite=False):
        """ Associates grasps in list grasps with the given object. Optionally associates the grasps with a single stable pose.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose
        force_overwrite : bool
            whether or not to overwrite grasps with the same id

        Returns
        -------
        bool
            True if grasps were stored for the given object and gripper, False otherwise
        """
        # create group for gripper if necessary
        if gripper not in list(self.grasp_data(key).keys()):
            self.grasp_data(key).create_group(gripper)
            self.grasp_data(key, gripper).attrs.create(NUM_GRASPS_KEY, 0)

        # store each grasp in the database
        return Hdf5ObjectFactory.write_grasps(grasps, self.grasp_data(key, gripper), force_overwrite)

    def grasp_metrics(self, key, grasps, gripper='pr2', stable_pose_id=None):
        """ Returns a list of grasp metric dictionaries fot the list of grasps provided to the database.

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metrics for
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose

        Returns
        -------
        :obj:`list` of :obj:`dict`
            dictionary mapping grasp ids to dictionaries that map metric names to numeric values
        """
        if gripper not in list(self.grasp_data(key).keys()):
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return {}
        return Hdf5ObjectFactory.grasp_metrics(grasps, self.grasp_data(key, gripper))

    def grasp_metric(self, key, grasp, metric_name, gripper, stable_pose_id=None):
        """ Return a single grasp metric, computing and storing if necessary. Not yet implemented.
        """
        raise NotImplementedError()

    def store_grasp_metrics(self, key, grasp_metric_dict, gripper='pr2', stable_pose_id=None, force_overwrite=False):
        """ Add grasp metrics in grasp_metric_dict to the data associated with grasps.

        Parameters
        ----------
        key : :obj:`str`
            key of object to store metrics for
        grasp_metric_dict : :obj:`dict` mapping :obj:`int` to :obj:`dict` mapping :obj:`str` to float
            mapping from grasp ids to a dictionary mapping metric names to numeric values
        gripper : :obj:`str`
            name of gripper
        stable_pose_id : :obj:`str`
            id of stable pose
        force_overwrite : bool
            whether or not to overwrite existing metrics with the same name

        Returns
        -------
        bool
            True if succcessful, False if aborted due to existing data
        """
        return Hdf5ObjectFactory.write_grasp_metrics(grasp_metric_dict, self.grasp_data(key, gripper), force_overwrite)

    # stable pose data
    def stable_poses(self, key, min_p=0.0):
        """ Stable poses for object key.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        min_p : float
            min stable pose probability to return

        Returns
        -------
        :obj:`list` of :obj:`StablePose`
            list of stable poses for the given object
        """
        stps = Hdf5ObjectFactory.stable_poses(self.stable_pose_data(key))

        # prune low probability stable poses
        stp_list = []
        for stp in stps:
            if stp.p > min_p:
                stp_list.append(stp)
        return stp_list

    def stable_pose(self, key, stable_pose_id):
        """ Stable pose of stable pose id for object key

        Parameters
        ----------
        key : :obj:`str`
            key of object
        stable_pose_id : obj:`str`
            id of stable pose to index

        Returns
        -------
        :obj:`StablePose`
            requested stable pose

        Raises
        ------
        ValueError
            If stable pose id is unrecognized
        """
        if stable_pose_id not in list(self.stable_pose_data(key).keys()):
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        return Hdf5ObjectFactory.stable_pose(self.stable_pose_data(key), stable_pose_id)

    # rendered image data
    def rendered_images(self, key, stable_pose_id=None, render_mode=RenderMode.DEPTH):
        """ Rendered images for the given object for the given render mode.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        stable_pose_id : :obj:`str`
            id of stable pose to index images for
        render_mode : :obj:`perception.RenderMode`
            modality of images to index (e.g. depth or segmask)

        Returns
        -------
        :obj:`list` of :obj:`perception.ObjectRender`
            list of stored images for the given object
        """
        if stable_pose_id is not None and stable_pose_id not in list(self.stable_pose_data(key).keys()):
            logging.warning('Stable pose id %s unknown' %(stable_pose_id))
            return[]
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in list(self.stable_pose_data(key)[stable_pose_id].keys()):
            logging.warning('No rendered images for stable pose %s' %(stable_pose_id))
            return []
        if stable_pose_id is not None and render_mode not in list(self.rendered_image_data(key, stable_pose_id).keys()):
            logging.warning('No rendered images of type %s for stable pose %s' %(render_mode, stable_pose_id))
            return []
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in list(self.object(key).keys()):
            logging.warning('No rendered images for object')
            return []
        if stable_pose_id is None and render_mode not in list(self.rendered_image_data(key).keys()):
            logging.warning('No rendered images of type %s for object' %(render_mode))
            return []

        rendered_images = Hdf5ObjectFactory.rendered_images(self.rendered_image_data(key, stable_pose_id, render_mode), render_mode=render_mode)
        for rendered_image in rendered_images:
            rendered_image.obj_key = key
        if stable_pose_id is not None:
            stable_pose = self.stable_pose(key, stable_pose_id)
            for rendered_image in rendered_images:
                rendered_image.stable_pose = stable_pose
        return rendered_images

    def has_rendered_images(self, key, stable_pose_id=None, render_mode=RenderMode.DEPTH):
        """ Checks whether or not a graspable has rendered images for the given stable pose and image type.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        stable_pose_id : :obj:`str`
            id of stable pose to index images for
        render_mode : :obj:`perception.RenderMode`
            modality of images to index (e.g. depth or segmask)

        Returns
        -------
        bool
            whether or not the dataset has images for the given pose and modality
        """
        if stable_pose_id is not None and stable_pose_id not in list(self.stable_pose_data(key).keys()):
            return False
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in list(self.stable_pose_data(key)[stable_pose_id].keys()):
            return False
        if stable_pose_id is not None and render_mode not in list(self.rendered_image_data(key, stable_pose_id).keys()):
            return False
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in list(self.object(key).keys()):
            return False
        if stable_pose_id is None and render_mode not in list(self.rendered_image_data(key).keys()):
            return False
        return True

    def delete_rendered_images(self, key, stable_pose_id=None, render_mode=RenderMode.DEPTH):
        """ Delete previously rendered images.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        stable_pose_id : :obj:`str`
            id of stable pose to index images for
        render_mode : :obj:`perception.RenderMode`
            modality of images to index (e.g. depth or segmask)

        Returns
        -------
        bool
            whether or not the images were deleted
        """
        if self.has_rendered_images(key, stable_pose_id, render_mode):
            del self.rendered_image_data(key, stable_pose_id)[render_mode]
            return True
        return False

    def store_rendered_images(self, key, rendered_images, stable_pose_id=None, render_mode=RenderMode.DEPTH, force_overwrite=False):
        """ Store rendered images of the object for a given stable pose.
        Parameters
        ----------
        key : :obj:`str`
            key of object
        rendered_images : :obj:`list` of :obj:`ObjectRender`
            list of images rendered for the given object
        stable_pose_id : obj:`str`
            id of stable pose to index images for
        render_mode : :obj:`perception.RenderMode`
            modality of images to index (e.g. depth or segmask)
        force_overwrite : bool
            True if existing images should be overwritten

        Returns
        -------
        bool
            whether or not the images were written
        """
        if stable_pose_id is not None and stable_pose_id not in list(self.stable_pose_data(key).keys()):
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in list(self.stable_pose_data(key)[stable_pose_id].keys()):
            self.stable_pose_data(key)[stable_pose_id].create_group(RENDERED_IMAGES_KEY)
        if stable_pose_id is not None and render_mode not in list(self.rendered_image_data(key, stable_pose_id).keys()):
            self.rendered_image_data(key, stable_pose_id).create_group(render_mode)
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in list(self.object(key).keys()):
            self.object(key).create_group(RENDERED_IMAGES_KEY)
        if stable_pose_id is None and render_mode not in list(self.rendered_image_data(key).keys()):
            self.rendered_image_data(key).create_group(render_mode)

        return Hdf5ObjectFactory.write_rendered_images(rendered_images, self.rendered_image_data(key, stable_pose_id, render_mode),
                                                             force_overwrite)

    def rendered_image_types(self, key, stable_pose_id=None):
        """ Return a list of the available rendered image modalities.

        Parameters
        ----------
        key : :obj:`str`
            key of object
        stable_pose_id : :obj:`str`
            id of stable pose to index images for

        Returns
        -------
        :obj:`list` of :obj:`str`
            list of available rendered image modes
        """
        if stable_pose_id is not None and stable_pose_id not in list(self.stable_pose_data(key).keys()):
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in list(self.stable_pose_data(key)[stable_pose_id].keys()):
            raise ValueError('No rendered images for stable pose %s' %(stable_pose_id))
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in list(self.object(key).keys()):
            raise ValueError('No rendered images for object %s' %(key))
        if stable_pose_id is None:
            return list(self.rendered_image_data(key).keys())

        render_modes = list(self.rendered_image_data(key, stable_pose_id).keys())
        return render_modes
    
    #connected components
    def store_connected_components(self, key, connected_components, force_overwrite=False):
        """ Store the connected components of the mesh
        
        Parameters
        ----------
        key : :obj:`str`
            key of object
        connected_components : :obj:`list` of :obj:`Mesh3D`
            connected components to write
        force_overwrite : bool
            True if existing connected components should be overwritten

        Returns
        -------
        bool
            whether or not the connected components were written
        """
        return Hdf5ObjectFactory.write_connected_components(connected_components, self.mesh_data(key), force_overwrite=force_overwrite)
        
    def connected_components(self, key):
        """ Returns a dict of the connected components of the given mesh. Returns None if connected components are unavailable.
        
        Parameters
        ----------
        key : :obj:`str`
            key of object
            
        Returns
        -------
        :obj:`dict` mapping :obj:`str` to :obj:`Mesh3D
            dict of connected components of the mesh. Keys are there for indexing.
        """
        return Hdf5ObjectFactory.connected_components(self.mesh_data(key))

    #metadata
    def create_metadata(self, metadata_name, metadata_type, metadata_description="No description"):
        """ Creates an object metadata with the given name for easier access.

        Parameters
        ----------
        metadata_name : :obj:`str`
            name of new metadata type
        metadata_type : :obj:`str`
            type of metadata. Must be {'scalar', 'array'}.
                primitive: anything that can be a non-object numpy scalar
                array:     numpy array with non-object dtype 
        metadata_description : :obj:`str`
            Description (helptext) for metadata. Optional

        Returns
        -------
        bool
            True if creation was successful, False otherwise
        """
        if metadata_type not in {'float', 'arr'}:
            logging.warning('metadata_type {} not supported'.format(metadata_type))
            return False
        
        # create metric data if nonexistent
        if self.metadata is None:
            self.data_.create_group(METADATA_KEY)
                        
        if self.has_metadata(metadata_name):
            logging.warning('Metadata %s already exists. Aborting...' %(metadata_name))
            return False
        self.metadata.create_group(metadata_name)

        # add description, type
        metadata_group = self.metadata_data(metadata_name)
        metadata_group.attrs.create(METADATA_TYPE_KEY, metadata_type)
        metadata_group.attrs.create(METADATA_DESC_KEY, metadata_description)
        return True
    
    def attach_metadata_func(self, metadata_name, metadata_func, overwrite=False, store_func=True):
        """ Attach a function that computes a given metadata from a Mesh3D object
        
        Parameters
        ----------
        metadata_name : :obj:`str`
            name of metadata type to attach function to
        metadata_func : :obj:`function`
            function that computes metric from Mesh3D object. Must return type consistent with what was set in create_metadata
            Note that for manually set metadata you can attach an object that isn't a function
        overwrite : boolean
            if True, overwrites existing metadata function
        store_func : boolean
            if True, attempts to use dill to serialize the function and store it in the database for later use.
            
        Returns
        -------
        bool
            True if succesful, False otherwise
        """
        if not self.has_metadata(metadata_name):
            logging.warning('metadata {} does not exist, attaching function failed'.format(metadata_name))
        if self.metadata_func(metadata_name) is not None and not overwrite:
            logging.warning('metadata {} already has a function attached and overwrite is False, aborting'.format(metadata_name))
        self._metadata_functions[metadata_name] = metadata_func
        if store_func:
            metadata_group = self.metadata_data(metadata_name)
            try:
                serialized = dill.dumps(metadata_func)
                serialized = serialized.ljust(len(serialized) + 32 - len(serialized) % 32, '\x00')
                serialized = np.fromstring(serialized, dtype=np.float32)
            except:
                logging.warning('Failed to serialize function. Aborting')
                return False
            if METADATA_FUNC_KEY in list(metadata_group.keys()):
                del metadata_group[METADATA_FUNC_KEY]
            metadata_group[METADATA_FUNC_KEY] = serialized
    
    def metadata_func(self, metadata_name):
        """ Returns function associated with metadata """
        if not self.has_metadata(metadata_name):
            logging.warning('metadata {} does not exist'.format(metadata_name))
            return None
        if metadata_name in list(self._metadata_functions.keys()):
            return self._metadata_functions[metadata_name]
        metadata_group = self.metadata_data(metadata_name)
        if METADATA_FUNC_KEY in list(metadata_group.keys()):
            return dill.loads(np.asarray(metadata_group[METADATA_FUNC_KEY]).tostring())
        return None
    
    def has_metadata(self, metadata_name):
        """ Checks if a metadata type already exists """
        return metadata_name in self.metadata_names

    def delete_metadata(self, metadata_name):
        """ Deletes a metadata type from the database.

        Parameters
        ----------
        metadata_name : :obj:`str`
            name of metadata to delete
        """
        if metadata_name in self.metadata_names:
            del self.metrics[metadata_name]
    
    def object_metadata(self, key):
        """ Returns a dictionary of object metadata for the object

        Parameters
        ----------
        key : :obj:`str`
            key of object to check metadata for
        metadata_type : :obj:`list` of :obj:`str`

        Returns
        -------
        :obj:`dict`
            dictionary mapping metadata names to numeric/numpy list values
        """
        return Hdf5ObjectFactory.object_metadata(self.mesh_data(key), self.metadata)
    
    def store_object_metadata(self, key, metadata_dict, force_overwrite=False):
        """ Manually write metadata

        Parameters
        ----------
        key : :obj:`str`
            key of object to store metadata for
        metadata_dict : :obj:`dict` mapping :obj:`str` to primatives/np arrays
            Dictionary of metadata to write. Maps metadata name to the metadata
        force_overwrite : bool
            whether or not to overwrite existing metadata with the same name

        Returns
        -------
        bool
            True if successful, False if aborted due to existing data
        """
        return Hdf5ObjectFactory.write_object_metadata(metadata_dict, self.mesh_data(key), self.get_metadata_types(), force_overwrite=force_overwrite)

    def compute_object_metadata(self, key, metadata_names=None, force_overwrite=False):
        """ Computes and stores metadata for object and metadata type(s) specified

        Parameters
        ----------
        key : :obj:`str`
            key of object to store metadata for
        metadata_name : :obj:`list` of :obj:`str`
            Name of metadata types to compute. if None computes all
        force_overwrite : bool
            whether or not to overwrite existing metadata with the same name

        Returns
        -------
        bool
            True if successful, False if aborted due to existing data
        """
        if metadata_names is None: metadata_names = self.metadata_names
        already_computed = set(self.object_metadata(key).keys()) & set(metadata_names)
        if len(already_computed) != 0 and not force_overwrite:
            logging.warning("Metadata already computed for object {} : {}".format(key, already_computed))
            return False
        metadata_dict = {}
        for metadata_name in metadata_names:
            logging.info("Computing metadata {} for object {}".format(metadata_name, key))
            metadata_func = self.metadata_func(metadata_name)
            if metadata_func is None:
                logging.warning("Metadata {} does not have an associated function, cannot compute".format(metadata_name))
                return False
            elif not callable(metadata_func):
                continue
            metadata_dict[metadata_name] = metadata_func(self.mesh(key))
        return self.store_object_metadata(key, metadata_dict, force_overwrite=force_overwrite)
        
    def get_metadata_desc(self, metadata_name):
        """ Returns description for metadata type
        
        Parameters
        ----------
        metadata_name : :obj:`str`
            Name of metadata to return description for
        """
        if not self.has_metadata(metadata_name):
            logging.warning('metadata {} does not exist'.format(metadata_name))
        return self.metadata_data(metadata_name).attrs[METADATA_DESC_KEY]
    
    def get_metadata_types(self):
        """ Returns a dictionary mapping metadata names to metadata types"""
        out = {}
        for metadata_name in self.metadata_names:
            out[metadata_name] = self.metadata_data(metadata_name).attrs[METADATA_TYPE_KEY]
        return out
