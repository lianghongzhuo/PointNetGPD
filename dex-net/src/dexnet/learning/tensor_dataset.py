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
Class to encapsulate tensorflow training datasets
Author: Jeff Mahler
"""
import IPython
import json
import logging
import numpy as np
import os
import sys

import autolab_core.utils as utils
from autolab_core import YamlConfig

from dexnet.constants import *

TENSOR_EXT = '.npy'
COMPRESSED_TENSOR_EXT = '.npz'

class Tensor(object):
    """ Abstraction for 4-D tensor objects. """
    def __init__(self, shape, dtype=np.float32):
        self.cur_index = 0
        self.dtype = dtype
        self.data = np.zeros(shape).astype(dtype)

    @property
    def shape(self):
        return self.data.shape

    @property
    def num_datapoints(self):
        return self.data.shape[0]

    @property
    def height(self):
        if len(self.data.shape) > 1:
            return self.data.shape[1]
        return None

    @property
    def width(self):
        if len(self.data.shape) > 2:
            return self.data.shape[2]
        return None

    @property
    def channels(self):
        if len(self.data.shape) > 3:
            return self.data.shape[3]
        return None

    @property
    def is_full(self):
        return self.cur_index == self.num_datapoints

    @property
    def has_data(self):
        return self.cur_index > 0

    def reset(self):
        """ Resets the current index. """
        self.cur_index = 0

    def add(self, datapoint):
        """ Adds the datapoint to the tensor if room is available. """
        if not self.is_full:
            self.set_datapoint(self.cur_index, datapoint)
            self.cur_index += 1

    def datapoint(self, ind):
        """ Returns the datapoint at the given index. """
        if self.height is None:
            return self.data[ind]
        return self.data[ind, ...]

    def set_datapoint(self, ind, datapoint):
        """ Sets the value of the datapoint at the given index. """
        self.data[ind, ...] = np.array(datapoint).astype(self.dtype)

    def data_slice(self, slice_ind):
        """ Returns a slice of datapoints """
        if self.data.height is None:
            return self.data[slice_ind]
        return self.data[slice_ind, ...]

    def save(self, filename, compressed=True):
        """ Save a tensor to disk. """
        # check for data
        if not self.has_data:
            return False

        # read ext and save accordingly
        _, file_ext = os.path.splitext(filename)
        if compressed:
            if file_ext != COMPRESSED_TENSOR_EXT:
                raise ValueError('Can only save compressed tensor with %s extension' %(COMPRESSED_TENSOR_EXT))
            np.savez_compressed(filename,
                                self.data[:self.cur_index,...])
        else:
            if file_ext != TENSOR_EXT:
                raise ValueError('Can only save tensor with .npy extension')
            np.save(filename, self.data[:self.cur_index,...])
        return True

    @staticmethod
    def load(filename, compressed=True):
        """ Loads a tensor from disk. """
        # switch load based on file ext
        _, file_ext = os.path.splitext(filename)
        if compressed:
            if file_ext != COMPRESSED_TENSOR_EXT:
                raise ValueError('Can only load compressed tensor with %s extension' %(COMPRESSED_TENSOR_EXT))
            data = np.load(filename)['arr_0']
        else:
            if file_ext != TENSOR_EXT:
                raise ValueError('Can only load tensor with .npy extension')
            data = np.load(filename)
        # init new tensor
        tensor = Tensor(data.shape, data.dtype)
        tensor.data = data.copy()
        return tensor

class TensorDatapoint(object):
    """ A single tensor datapoint.
    Basically acts like a dictionary
    """
    def __init__(self, field_names):
        self._data = {}
        for field_name in field_names:
            self._data[field_name] = None

    def __getitem__(self, name):
        """ Return a data field. """
        return self._data[name]

    def __setitem__(self, name, value):
        """ Set a data field. """
        self._data[name] = value
            
class TensorDataset(object):
    """ Encapsulates learning datasets and different training and test
    splits of the data. """
    def __init__(self, filename, config, access_mode=WRITE_ACCESS):
        # read params
        self._filename = filename
        self._config = config
        self._datapoints_per_file = config['datapoints_per_file']
        self._access_mode = access_mode
 
        # check valid access mode
        if access_mode == READ_WRITE_ACCESS:
            raise ValueError('Read and write not supported simultaneously.')
       
        # open dataset folder
        # create dataset if necessary
        if not os.path.exists(self._filename) and access_mode != READ_ONLY_ACCESS:
            os.mkdir(self._filename)
        # throw error if dataset doesn't exist
        elif not os.path.exists(self._filename) and access_mode == READ_ONLY_ACCESS:
            raise ValueError('Dataset %s does not exist!' %(self._filename))
        # check dataset empty
        elif os.path.exists(self._filename) and len(os.listdir(self._filename)) > 0 and access_mode == WRITE_ACCESS:
            human_input = utils.keyboard_input('Dataset %s exists. Overwrite?' %(self.filename), yesno=True)
            if human_input.lower() == 'n':
                raise ValueError('User opted not to overwrite dataset')

        # save config to location
        if access_mode != READ_ONLY_ACCESS:
            config_filename = os.path.join(self._filename, 'config.json')
            json.dump(self._config, open(config_filename, 'w'))

        # init data storage
        self._allocate_tensors()

        # init state variables
        if access_mode == WRITE_ACCESS:
            # init no files
            self._num_tensors = 0
            self._num_datapoints = 0
            if not os.path.exists(self.tensor_dir):
                os.mkdir(self.tensor_dir)

        elif access_mode == READ_ONLY_ACCESS:
            # read the number of tensor files
            tensor_dir = self.tensor_dir
            tensor_filenames = utils.filenames(tensor_dir, tag=COMPRESSED_TENSOR_EXT, sorted=True)
            file_nums = np.array([int(filename[-9:-4]) for filename in tensor_filenames])

            self._num_tensors = np.max(file_nums)+1

            # compute the number of datapoints
            last_tensor_ind = np.where(file_nums == self._num_tensors-1)[0][0]
            last_tensor_data = np.load(tensor_filenames[last_tensor_ind])['arr_0']
            self._num_datapoints_last_file = last_tensor_data.shape[0]
            self._num_datapoints = self._datapoints_per_file * (self._num_tensors-1) + self._num_datapoints_last_file

            # form index maps for each file
            self._index_to_file_num = {}
            self._file_num_to_indices = {}

            # set file index
            cur_file_num = 0
            start_datapoint_index = 0

            # set mapping from file num to datapoint indices
            self._file_num_to_indices[cur_file_num] = np.arange(self._datapoints_per_file) + start_datapoint_index

            for ind in range(self._num_datapoints):
                # set mapping from index to file num
                self._index_to_file_num[ind] = cur_file_num

                # update to the next file
                if ind > 0 and ind % self._datapoints_per_file == 0:
                    cur_file_num += 1
                    start_datapoint_index += self._datapoints_per_file

                    # set mapping from file num to datapoint indices
                    if cur_file_num < self._num_tensors-1:
                        self._file_num_to_indices[cur_file_num] = np.arange(self._datapoints_per_file) + start_datapoint_index
                    else:
                        self._file_num_to_indices[cur_file_num] = np.arange(self._num_datapoints_last_file) + start_datapoint_index
        else:
            raise ValueError('Access mode %s not supported' %(access_mode))

    @property
    def filename(self):
        return self._filename

    @property
    def config(self):
        return self._config
        
    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def num_datapoints(self):
        return self._num_datapoints

    @property
    def datapoints_per_file(self):
        return self._datapoints_per_file

    @property
    def field_names(self):
        return list(self._tensors.keys())

    @property
    def datapoint_template(self):
        return TensorDatapoint(self.field_names)

    @property
    def datapoint_indices(self):
        """ Returns an array of all dataset indices. """
        return np.arange(self._num_datapoints)

    @property
    def tensor_indices(self):
        """ Returns an array of all tensor indices. """
        return np.arange(self._num_tensors)
    
    @property
    def tensor_dir(self):
        """ Return the tensor directory. """
        return os.path.join(self._filename, 'tensors')

    def datapoint_indices_for_tensor(self, tensor_index):
        """ Returns the indices for all datapoints in the given tensor. """
        if tensor_index >= self._num_tensors:
            raise ValueError('Tensor index %d is greater than the number of tensors (%d)' %(tensor_index, self._num_tensors))
        return self._file_num_to_index[tensor_index]

    def tensor_index(self, datapoint_index):
        """ Returns the index of the tensor containing the referenced datapoint. """
        if datapoint_index >= self._num_datapoints:
            raise ValueError('Datapoint index %d is greater than the number of datapoints (%d)' %(datapoint_index, self._num_datapoints))
        return self._index_to_file_num[datapoint_index]

    def generate_tensor_filename(self, field_name, file_num, compressed=True):
        """ Generate a filename for a tensor. """
        file_ext = TENSOR_EXT
        if compressed:
            file_ext = COMPRESSED_TENSOR_EXT
        filename = os.path.join(self.filename, 'tensors', '%s_%05d%s' %(field_name, file_num, file_ext))
        return filename

    def _allocate_tensors(self):
        """ Allocates the tensors in the dataset. """
        # init tensors dict
        self._tensors = {}

        # allocate tensor for each data field
        for field_name, field_spec in self._config['fields'].items():
            # parse attributes
            field_dtype = np.dtype(field_spec['dtype'])
            
            # parse shape
            field_shape = [self._datapoints_per_file]
            if 'height' in list(field_spec.keys()):
                field_shape.append(field_spec['height'])
                if 'width' in list(field_spec.keys()):
                    field_shape.append(field_spec['width'])
                    if 'channels' in list(field_spec.keys()):
                        field_shape.append(field_spec['channels'])
                        
            # create tensor
            self._tensors[field_name] = Tensor(field_shape, field_dtype)

    def add(self, datapoint):
        """ Adds a datapoint to the file. """
        # check access level
        if self._access_mode == READ_ONLY_ACCESS:
            raise ValueError('Cannot add datapoints with read-only access')

        # store data in tensor
        for field_name in self.field_names:
            self._tensors[field_name].add(datapoint[field_name])

        # save if tensors are full
        field_name = self.field_names[0]
        if self._tensors[field_name].is_full:
            # save next tensors to file
            self.write()

        # increment num datapoints
        self._num_datapoints += 1

    def __getitem__(self, ind):
        """ Indexes the dataset for the datapoint at the given index. """
        return self.datapoint(ind)

    def datapoint(self, ind):
        """ Loads a tensor datapoint for a given global index.

        Parameters
        ----------
        ind : int
            global index in the tensor

        Returns
        -------
        :obj:`TensorDatapoint`
            the desired tensor datapoint
        """
        # check valid input
        if ind >= self._num_datapoints:
            raise ValueError('Index %d larger than the number of datapoints in the dataset (%d)' %(ind, self._num_datapoints))

        # return the datapoint
        datapoint = self.datapoint_template
        file_num = self._index_to_file_num[ind]
        for field_name in self.field_names:
            tensor = self.load_tensor(field_name, file_num)
            tensor_index = ind % self._datapoints_per_file
            datapoint[field_name] = tensor.datapoint(tensor_index)
        return datapoint

    def load_tensor(self, field_name, file_num):
        """ Loads a tensor for a given field and file num.

        Parameters
        ----------
        field_name : str
            the name of the field to load
        file_num : int
            the number of the file to load from

        Returns
        -------
        :obj:`Tensor`
            the desired tensor
        """
        filename = self.generate_tensor_filename(field_name, file_num, compressed=True)
        tensor = Tensor.load(filename, compressed=True)
        return tensor

    def __iter__(self):
        """ Generate iterator. Not thread safe. """
        self._count = 0
        return self

    def __next__(self):
        """ Read the next datapoint.
        
        Returns
        -------
        :obj:`TensorDatapoint`
            the next datapoint
        """
        # terminate
        if self._count >= self._num_datapoints:
            raise StopIteration

        # init empty datapoint
        datapoint = self.datapoint(self._count)
        self._count += 1
        return datapoint

    def write(self):
        """ Writes all tensors to the next file number. """
        # write the next file for all fields
        for field_name in self.field_names:
            filename = self.generate_tensor_filename(field_name, self._num_tensors)
            self._tensors[field_name].save(filename, compressed=True)
            self._tensors[field_name].reset()
        self._num_tensors += 1

    def flush(self):
        """ Flushes the data tensors. Alternate handle to write. """
        self.write()

    @staticmethod
    def open(dataset_dir):
        """ Opens a tensor dataset. """
        # read config
        config_filename = os.path.join(dataset_dir, 'config.json')
        config = json.load(open(config_filename, 'r'))

        # open dataset
        dataset = TensorDataset(dataset_dir, config, access_mode=READ_ONLY_ACCESS)
        return dataset
        
    def split(self, attribute, train_pct, val_pct):
        """ Splits the dataset along the given attribute. """
        # determine valid values of attribute
        
        # split on values

        # find indices corresponding to split values

        # save split

        raise NotImplementedError()
