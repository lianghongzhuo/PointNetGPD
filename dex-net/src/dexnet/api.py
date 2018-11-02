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
import collections
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tempfile
import time

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from autolab_core import YamlConfig, RigidTransform

# TODO
# Once trimesh integration is here via meshpy remove this
import trimesh

import dexnet.database.database as db
import dexnet.grasping.grasp_quality_config as gqc
import dexnet.grasping.grasp_quality_function as gqf
import dexnet.grasping.grasp_sampler as gs
import dexnet.grasping.gripper as gr
import dexnet.database.mesh_processor as mp
from meshpy import convex_decomposition, Mesh3D

try:
    from dexnet.visualization.visualizer3d import DexNetVisualizer3D as vis
except:
    logger.warning('Failed to import DexNetVisualizer3D, visualization methods will be unavailable')

DEXNET_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../../') + '/'
# test add by liang
# import dexnet.visualization.visualizer3d
DEXNET_API_DEFAULTS_FILE = DEXNET_DIR + 'cfg/api_defaults.yaml'


class DexNet(object):
    """Class providing an interface for main DexNet pipeline
    
    Attributes
    ----------
    database : :obj:`dexnet.database.Database
        Current active database. Can set manually, or with open_database
    dataset : :obj:`dexnet.database.Dataset
        Current active dataset. Can set manually, or with open_dataset
    default_config : :obj:`dictionary`
        A dictionary containing default config values
        See Other Parameters for details. These parameters are also listed under the function(s) they are relevant to
        Also, see dexnet.database.mesh_processor for configs associated with initial mesh processing
        
    Other Parameters
    ----------------
    cache_dir 
        Cache directory for to store intermediate files. If None uses a temporary directory
    gripper_dir
        Directory where the grippers models and parameters are
    export_format
        Format for export. One of obj, stl, urdf
    export_scale
        Scale for export.
    export_overwrite
        If True, will overwrite existing files
    animate
        Whether or not to animate the displayed object
    quality_scale
        Range to scale quality metric values to
    show_gripper
        Whether or not to show the gripper in the visualization
    min_metric
        lowest value of metric to show grasps for
    max_plot_gripper
        Number of grasps to plot
    """

    def __init__(self):
        """Create a DexNet object
        """
        self.database = None
        self.dataset = None

        self._database_temp_cache_dir = None

        # open default config
        self.default_config = YamlConfig(DEXNET_API_DEFAULTS_FILE)
        # Resolve gripper_dir and cache_dir relative to dex-net root
        for key in ['gripper_dir', 'cache_dir']:
            if not os.path.isabs(self.default_config[key]):
                self.default_config[key] = os.path.realpath(DEXNET_DIR + self.default_config[key])

    # TODO
    # Move to YamlConfig
    @staticmethod
    def _deep_update_config(config, updates):
        """ Deep updates a config dict """
        for key, value in updates.items():
            if isinstance(value, collections.Mapping):
                config[key] = DexNet._deep_update_config(config.get(key, {}), value)
            else:
                config[key] = value
        return config

    def _get_config(self, updates=None):
        """ Gets a copy of the default config dict with updates from the dict passed in applied """
        updated_cfg = copy.deepcopy(self.default_config.config)
        if updates is not None:
            DexNet._deep_update_config(updated_cfg, updates)
        return updated_cfg

    def _check_opens(self):
        """ Checks that database and dataset are open """
        if self.database is None:
            raise RuntimeError('You must open a database first')
        if self.dataset is None:
            raise RuntimeError('You must open a dataset first')

    def open_database(self, database_path, config=None, create_db=True):
        """Open/create a database.

        Parameters
        ----------
        database_path : :obj:`str`
            Path (can be relative) to the database, or the path to create a database at.        
        create_db : boolean
            If True, creates database if one does not exist at location specified.
            If False, raises error if database does not exist at location specified.
        config : :obj:`dict`
            Dictionary of parameters for database creation
            Parameters are in Other Parameters. Values from self.default_config are used for keys not provided.
            
        Other Parameters
        ----------------
        cache_dir 
            Cache directory for to store intermediate files. If None uses a temporary directory
            
        Raises
        ------
        ValueError
            If database_path does not have an extension corresponding to a hdf5 database.
            If database does not exist at path and create_db is False.
        """
        config = self._get_config(config)

        if self.database is not None:
            if self._database_temp_cache_dir is not None:
                shutil.rmtree(self._database_temp_cache_dir)
                self._database_temp_cache_dir = None
            self.database.close()

        # Check database path extension
        _, database_ext = os.path.splitext(database_path)
        if database_ext != db.HDF5_EXT:
            raise ValueError('Database must have extension {}'.format(db.HDF5_EXT))

            # Abort if database does not exist and create_db is False
        if not os.path.exists(database_path):
            if not create_db:
                raise ValueError('Database does not exist at path {} and create_db is False'.format(database_path))
            else:
                logger.info("File not found, creating new database at {}".format(database_path))

        # Create temp dir if cache dir is not provided
        cache_dir = config['cache_dir']
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp()
            self._database_temp_cache_dir = cache_dir

        # Open database
        self.database = db.Hdf5Database(database_path,
                                        access_level=db.READ_WRITE_ACCESS,
                                        cache_dir=cache_dir)

    def open_dataset(self, dataset_name, config=None, create_ds=True):
        """Open/create a dataset

        Parameters
        ----------
        dataset_name : :obj:`str`
            Name of dataset to open/create
        create_ds : boolean
            If True, creates dataset if one does not exist with name specified.
            If False, raises error if specified dataset does not exist
        config : :obj:`dict`
            Dictionary containing a key 'metrics' that maps to a dictionary mapping metric names to metric config dicts
            For available metrics and their corresponding config parameters see dexnet.grasping.grasp_quality_config
            Values from self.default_config are used for keys not provided
        
        Raises
        ------
        ValueError
            If dataset_name is invalid. Also if dataset does not exist and create_ds is False
        RuntimeError
            No database open
        """
        if self.database is None:
            raise RuntimeError('You must open a database first')

        config = self._get_config(config)

        tokens = dataset_name.split()
        if len(tokens) > 1:
            raise ValueError("dataset_name \"{}\" is invalid (contains delimiter)".format(dataset_name))

        existing_datasets = [d.name for d in self.database.datasets]

        # create/open new ds
        if dataset_name not in existing_datasets:
            if create_ds:
                logger.info("Creating new dataset {}".format(dataset_name))
                self.database.create_dataset(dataset_name)
                self.dataset = self.database.dataset(dataset_name)
                metric_dict = config['metrics']
                for metric_name, metric_spec in metric_dict.items():
                    # create metric
                    metric_config = gqc.GraspQualityConfigFactory.create_config(metric_spec)
                    self.dataset.create_metric(metric_name, metric_config)
            else:
                raise ValueError(
                    "dataset_name \"{}\" is invalid (does not exist, and create_ds is False)".format(dataset_name))
        else:
            self.dataset = self.database.dataset(dataset_name)

        if self.dataset.metadata is None:
            self._attach_metadata()

    # TODO
    # Once trimesh integration is here via meshpy remove this
    @staticmethod
    def _meshpy_to_trimesh(mesh_m3d):
        vertices = mesh_m3d.vertices
        faces = mesh_m3d.triangles
        mesh_tm = trimesh.Trimesh(vertices, faces)
        return mesh_tm

    # TODO
    # Once trimesh integration is here via meshpy remove this
    @staticmethod
    def _trimesh_to_meshpy(mesh_tm):
        vertices = mesh_tm.vertices
        triangles = mesh_tm.faces
        mesh_m3d = Mesh3D(vertices, triangles)
        return mesh_m3d

    # TODO
    # Once trimesh integration is here via meshpy remove this
    @staticmethod
    def is_watertight(mesh):
        mesh_tm = DexNet._meshpy_to_trimesh(mesh)
        return mesh_tm.is_watertight

    # TODO
    # Make this better and more general
    def _attach_metadata(self):
        """ Attach default metadata to dataset. Currently only watertightness and number of connected components, and
        only watertightness has an attached function.
        """
        self.dataset.create_metadata("watertightness", "float", "1.0 if the mesh is watertight, 0.0 if it is not")
        self.dataset.attach_metadata_func("watertightness", DexNet.is_watertight, overwrite=False, store_func=True)
        self.dataset.create_metadata("num_con_comps", "float",
                                     "Number of connected components (may not be watertight) in the mesh")
        self.dataset.attach_metadata_func("num_con_comps", object(), overwrite=False, store_func=True)

    @staticmethod
    def _single_obj_grasps(dataset, obj, gripper, config, stable_pose_id=None):
        """ Sample grasps and compute metrics for given object, gripper, and stable pose """

        # create grasp sampler
        logger.info('Sampling grasps')
        if config['grasp_sampler'] == 'antipodal':
            sampler = gs.AntipodalGraspSampler(gripper, config)
        elif config['grasp_sampler'] == 'gaussian':
            sampler = gs.GaussianGraspSampler(gripper, config)
        elif config['grasp_sampler'] == 'uniform':
            sampler = gs.UniformGraspSampler(gripper, config)

        # sample grasps
        grasps = sampler.generate_grasps(obj, max_iter=config['max_grasp_sampling_iters'])
        return grasps

    def compute_simulation_data(self, object_name, config=None):
        """Compute normals and convex decomposition for object (preprocessing for simulation)
        
        Parameters
        ----------
        object_name
            Object to compute normals and convex decomposition for
        config : :obj:`dict`
            Configuration dict for computing simulation data.\
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
            
        Other Parameters
        -----------------
        cache_dir 
            Cache directory for to store intermediate files. If None uses a temporary directory
                    
        Raises
        ------
        RuntimeError
            Database or dataset not opened.
        """
        self._check_opens()
        config = self._get_config(config)

        # Create temp dir if cache dir is not provided
        cache_dir = config['cache_dir']
        del_cache = False
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp()
            del_cache = True

        obj = self.dataset[object_name]
        if obj.mesh.normals is None:
            logger.info('Computing vertex normals for {}'.format(object_name))
            obj.mesh.compute_vertex_normals()
            self.dataset.store_mesh(object_name, obj.mesh, force_overwrite=True)
        logger.info('Running convex decomposition for {}'.format(object_name))
        try:
            convex_pieces, _, _ = convex_decomposition(obj.mesh, cache_dir=cache_dir, name=object_name)
        except Exception as e:
            logging.error('Convex decomposition failed. Did you install v-hacd?')
            raise e
        self.dataset.delete_convex_pieces(object_name)
        self.dataset.store_convex_pieces(object_name, convex_pieces)

        if del_cache:
            shutil.rmtree(cache_dir)

    def compute_metadata(self, object_name, config=None, overwrite=False):
        """Compute metadata for object
        
        Parameters
        ----------
        object : :obj:`str`
            Object name to compute metadata for
        overwrite : boolean
            If True, overwrites existing metadata. Otherwise, logs a warning and keeps existing metadata 
            
        Raises
        ------
        RuntimeError
            Database or dataset not opened.
        """
        self._check_opens()
        config = self._get_config(config)
        self.dataset.compute_object_metadata(object_name, force_overwrite=overwrite)
        if (not overwrite and self.dataset.connected_components(object_name) is not None
                and 'num_con_comps' in list(self.dataset.object_metadata(
                    object_name).keys())):  # Remove static references to num_con_comps
            raise RuntimeError("Connected components data already exists for object {}, aborting".format(object_name))

        # TODO
        # Fix this once trimesh functionality is integrated into meshpy
        ccs_trm = DexNet._meshpy_to_trimesh(self.dataset.mesh(object_name)).split(only_watertight=False)
        ccs_m3d = []
        for cc in ccs_trm:
            ccs_m3d.append(DexNet._trimesh_to_meshpy(cc))
        self.dataset.store_object_metadata(object_name, {"num_con_comps": len(ccs_m3d)})
        self.dataset.store_connected_components(object_name, ccs_m3d)

    def get_metadata(self, object_name, config=None):
        """Get metadata for object
        
        Parameters
        ----------
        object_name : :obj:`str`
            object name to get metadata for
            
        Raises
        ------
        RuntimeError
            Database or dataset not opened.
        """
        self._check_opens()
        config = self._get_config(config)

        return self.dataset.object_metadata(object_name)

    def export_objects(self, output_dir, export_filters={}, to_export=None, config=None):
        """Export objects as .obj files to a directory. Provides filtering ability to only export some objects
        
        Parameters
        ----------
        output_dir : :obj:`str`
            Directory to output to objects to
        export_filters : :obj:`dict` mapping :obj:`str` to :obj:function
            Functions to filter with. Each function takes in the metadata with its key as the key associated with each
            object and returns True or False. If True exports object, if False doesn't.
            Example: {'num_con_comps' : (lambda x: x == 1)} will export only objects with exactly one connected component
        to_export : :obj:`list` of :obj:`str`
            List of objects to export. If None exports all objects in dataset
        config : :obj:`dict`
            Configuration dict for computing simulation data.
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
        
        Other Parameters
        ----------------
        export_format
            Format for export. One of obj, stl, urdf
        export_scale
            Scale for export.
        export_overwrite
            If True, will overwrite existing files
        
        Raises
        ------
        RuntimeError
            Database or dataset not opened.
        ValueError
            Export format not supported
        """
        self._check_opens()
        config = self._get_config(config)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if to_export is None:
            to_export = self.list_objects()
        for object_name in to_export:
            metadata_dict = self.get_metadata(object_name, config=config)
            pass_filters = True
            for metadata_name, filter_fn in export_filters.items():
                if metadata_name not in list(metadata_dict.keys()):
                    logger.warning(
                        "Metadata {} not computed for object {}! Excluding object.".format(metadata_name, object_name))
                    pass_filters = False
                    break
                pass_filters = pass_filters & filter_fn(metadata_dict[metadata_name])
            if pass_filters:
                if config["export_format"] == 'obj':
                    self.dataset.obj_mesh_filename(object_name, scale=config["export_scale"], output_dir=output_dir,
                                                   overwrite=config["export_overwrite"])
                elif config["export_format"] == 'stl':
                    self.dataset.stl_mesh_filename(object_name, scale=config["export_scale"], output_dir=output_dir,
                                                   overwrite=config["export_overwrite"])
                elif config["export_format"] == 'urdf':
                    self.dataset.stl_mesh_filename(object_name, scale=config["export_scale"], output_dir=output_dir,
                                                   overwrite=config["export_overwrite"])
                else:
                    raise ValueError("Export format {} not supported".format(config["export_format"]))

    def list_grippers(self, config=None):
        """List available grippers
        
        Parameters
        ----------
        config : :obj:`dict`
            Configuration dict.
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
            
        Other Parameters
        ----------------
        gripper_dir
            Directory where the grippers models and parameters are
        """
        config = self._get_config(config)
        gripper_dir = config['gripper_dir']
        return [gripper for gripper in os.listdir(gripper_dir) if os.path.isdir(os.path.join(gripper_dir, gripper))]

    def list_metrics(self, config=None):
        """List available metrics
        
        Parameters
        ----------
        config : :obj:`dict`
            Configuration dict.
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
            
        Other Parameters
        ----------------
        metrics
            Dictionary mapping metric names to metric config dicts
            
        Returns
        -------
        :obj:`list` of :obj:`str`
            List of metric names
        """
        config = self._get_config(config)
        return list(config['metrics'].keys())

    def list_metadata(self):
        """List available metadata names.
            
        Returns
        -------
        :obj:`list` of :obj:`str`
            List of metadata names
        """
        return self.dataset.metadata_names

    def list_objects(self):
        """List available objects in current dataset
            
        Returns
        -------
        :obj:`list` of :obj:`str`
            List of objects in current dataset
        """
        self._check_opens()
        return self.dataset.object_keys

    def get_object(self, object_name):
        """Get an object from current dataset by name
        
        Parameters
        ----------
        object_name : :obj:`str`
            Name of object to get
            
        Returns
        -------
        :obj:`Mesh3D`
            Specified object
        """
        self._check_opens()
        return self.dataset[object_name].mesh

    def get_stable_poses(self, object_name):
        """Get stable poses for an object by name
        
        Parameters
        ----------
        object_name : :obj:`str`
            Name of object to get
        
        Returns
        ------
        :obj:`list` of :obj:`meshpy.StablePose`
            Stable poses of object
        """
        self._check_opens()
        return self.dataset.stable_poses(object_name)

    def get_grasps(self, object_name, gripper_name, metric_name=None):
        """ Returns the list of grasps for the given graspable sorted by decreasing quality according to the given metric.

        Parameters
        ----------
        object_name : :obj:`str`
            name of object to get grasps for
        gripper_name : :obj:`str`
            name of gripper
        metric_name : :obj:`str`
            name of metric to use for sorting. If None does not sort grasps

        Returns
        -------
        :obj:`list` of :obj:`dexnet.grasping.ParallelJawPtGrasp3D`
            stored grasps for the object and gripper sorted by metric in descending order, empty list if gripper not found
        :obj:`list` of float
            values of metrics for the grasps sorted in descending order, empty list if gripper not found or if metric_name not given
        """
        self._check_opens()
        if metric_name is None:
            return self.dataset.grasps(object_name, gripper=gripper_name)
        return self.dataset.sorted_grasps(object_name, metric_name, gripper=gripper_name)

    def display_object(self, object_name, config=None):
        """Display an object
        
        Parameters
        ----------
        object_name : :obj:`str`
            Object to display.
        config : :obj:`dict`
            Configuration dict for visualization.
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
        
        Other Parameters
        ----------------
        animate
            Whether or not to animate the displayed object
            
        Raises
        ------
        ValueError
            invalid object name
        RuntimeError
            Database or dataset not opened.
        """
        self._check_opens()
        config = self._get_config(config)

        if object_name not in self.dataset.object_keys:
            raise ValueError("{} is not a valid object name".format(object_name))

        logger.info('Displaying {}'.format(object_name))
        obj = self.dataset[object_name]

        vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
        vis.mesh(obj.mesh.trimesh, color=(0.5, 0.5, 0.5), style='surface')
        vis.show(animate=config['animate'])

    def display_stable_poses(self, object_name, config=None):
        """Display an object's stable poses
        
        Parameters
        ----------
        object_name : :obj:`str`
            Object to display.
        config : :obj:`dict`
            Configuration dict for visualization.
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
        
        Other Parameters
        ----------------
        animate
            Whether or not to animate the displayed object
            
        Raises
        ------
        ValueError
            invalid object name
        RuntimeError
            Database or dataset not opened.
        """
        self._check_opens()
        config = self._get_config(config)

        if object_name not in self.dataset.object_keys:
            raise ValueError("{} is not a valid object name".format(object_name))

        logger.info('Displaying stable poses for'.format(object_name))
        obj = self.dataset[object_name]
        stable_poses = self.dataset.stable_poses(object_name)

        for stable_pose in stable_poses:
            # print 'Stable pose %s with p=%.3f' %(stable_pose.id, stable_pose.p)
            vis.figure()
            vis.mesh_stable_pose(obj.mesh.trimesh, stable_pose.T_obj_world,
                                 color=(0.5, 0.5, 0.5), style='surface')
            vis.pose(RigidTransform(), alpha=0.15)
            vis.show(animate=config['animate'])

    def display_grasps(self, object_name, gripper_name, metric_name, config=None):
        """ Display grasps for an object 
        
        Parameters
        ----------
        object_name : :obj:`str`
            Object to display
        gripper_name : :obj:`str`
            Gripper for which to display grasps
        metric_name : :obj:`str`
            Metric to color/rank grasps with
        config : :obj:`dict`
            Configuration dict for visualization. 
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
        
        Other Parameters
        ----------------
        gripper_dir
            Directory where the grippers models and parameters are.
        quality_scale
            Range to scale quality metric values to
        show_gripper
            Whether or not to show the gripper in the visualization
        min_metric
            lowest value of metric to show grasps for
        max_plot_gripper
            Number of grasps to plot
        animate
            Whether or not to animate the displayed object
        """
        self._check_opens()
        config = self._get_config(config)

        grippers = os.listdir(config['gripper_dir'])
        if gripper_name not in grippers:
            raise ValueError("{} is not a valid gripper name".format(gripper_name))
        gripper = gr.RobotGripper.load(gripper_name, gripper_dir=config['gripper_dir'])

        objects = self.dataset.object_keys
        if object_name not in objects:
            raise ValueError("{} is not a valid object name".format(object_name))

        metrics = self.dataset.available_metrics(object_name, gripper=gripper.name)
        if metric_name not in metrics:
            raise ValueError(
                "{} is not computed for gripper {}, object {}".format(metric_name, gripper.name, object_name))

        logger.info('Displaying grasps for gripper %s on object %s' % (gripper.name, object_name))
        object = self.dataset[object_name]
        grasps, metrics = self.dataset.sorted_grasps(object_name, metric_name,
                                                     gripper=gripper.name)

        if len(grasps) == 0:
            raise RuntimeError('No grasps for gripper %s on object %s' % (gripper.name, object_name))
            return

        low = np.min(metrics)
        high = np.max(metrics)
        if low == high:
            q_to_c = lambda quality: config['quality_scale']
        else:
            q_to_c = lambda quality: config['quality_scale'] * (quality - low) / (high - low)

        if config['show_gripper']:
            i = 0
            stable_pose = self.dataset.stable_pose(object.key, 'pose_1')
            for grasp, metric in zip(grasps, metrics):
                if metric <= config['min_metric']:
                    continue

                    # print 'Grasp %d %s=%.5f' %(grasp.id, metric_name, metric)
                T_obj_world = RigidTransform(from_frame='obj',
                                             to_frame='world')
                color = plt.get_cmap('hsv')(q_to_c(metric))[:-1]
                T_obj_gripper = grasp.gripper_pose(gripper)
                grasp = grasp.perpendicular_table(stable_pose)
                vis.figure()
                vis.gripper_on_object(gripper, grasp, object,
                                      gripper_color=(0.25, 0.25, 0.25),
                                      stable_pose=stable_pose.T_obj_world,
                                      plot_table=False)
                vis.show(animate=config['animate'])
                i += 1
                if i >= config['max_plot_gripper']:
                    break
        else:
            i = 0
            vis.figure()
            vis.mesh(object.mesh.trimesh, style='surface')
            for grasp, metric in zip(grasps, metrics):
                if metric <= config['min_metric']:
                    continue

                    # print 'Grasp %d %s=%.5f' %(grasp.id, metric_name, metric)
                T_obj_world = RigidTransform(from_frame='obj',
                                             to_frame='world')
                color = plt.get_cmap('hsv')(q_to_c(metric))[:-1]
                T_obj_gripper = grasp.gripper_pose(gripper)
                vis.grasp(grasp, grasp_axis_color=color,
                          endpoint_color=color)
                i += 1
                if i >= config['max_plot_gripper']:
                    break

            vis.show(animate=config['animate'])

    def delete_object(self, object_name):
        """ Delete an object
        
        Parameters
        ----------
        object_name : :obj:`str`
            Object to delete
        
        Raises
        ------
        ValueError
            invalid object name
        RuntimeError
            Database or dataset not opened
        """
        self._check_opens()
        if object_name not in self.dataset.object_keys:
            raise ValueError("{} is not a valid object name".format(object_name))

        logger.info('Deleting {}'.format(object_name))
        self.dataset.delete_graspable(object_name)

    def close_database(self):
        if self.database:
            logger.info('Closing database')
            self.database.close()
        # Delete cache if using temp cache
        if self._database_temp_cache_dir is not None:
            shutil.rmtree(self._database_temp_cache_dir)
