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
Command line tool for using database
Author: Jeff Mahler
"""
import argparse
import collections
import IPython
import logging
import os
import re
import readline
import signal
import dexnet

DEFAULT_CONFIG = 'cfg/apps/cli_parameters.yaml'
SUPPORTED_MESH_FORMATS = ['.obj', '.off', '.wrl', '.stl']
RE_SPACE = re.compile('.*\s+$', re.M)
MAX_QUEUE_SIZE = 1000

logging.root.name = 'dex-net'


class Completer(object):
    """
    Tab completion class for Dex-Net CLI.
    Adapted from http://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input
    """

    def __init__(self, commands=[]):
        """ Provide a list of commands """
        self.commands = commands

        # dexnet entity tab completion
        self.words = []

    def _listdir(self, root):
        "List directory 'root' appending the path separator to subdirs."
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path=None):
        "Perform completion of filesystem path."
        if path is None or path == '':
            return self._listdir('./')
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else '.'
        res = [os.path.join(dirname, p)
               for p in self._listdir(tmp) if p.startswith(rest)]
        # more than one match, or single match which does not exist (typo)
        if len(res) > 1 or not os.path.exists(path):
            return res
        # resolved to a single directory, so return list of files below it
        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]
        # exact file match terminates this completion
        return [path + ' ']

    def complete_extra(self, args):
        "Completions for the 'extra' command."
        # treat the last arg as a path and complete it
        if len(args) == 0:
            return self._listdir('./')
        return self._complete_path(args[-1])

    def complete(self, text, state):
        "Generic readline completion entry point."

        # dexnet entity tab completion
        results = [w for w in self.words if w.startswith(text)] + [None]
        if results != [None]:
            return results[state]

        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()

        # dexnet entity tab completion
        results = [w for w in self.words if w.startswith(text)] + [None]
        if results != [None]:
            return results[state]

        # account for last argument ending in a space
        if RE_SPACE.match(buffer):
            line.append('')

        return (self.complete_extra(line) + [None])[state]

    # dexnet entity tab completion
    def set_words(self, words):
        self.words = [str(w) for w in words]


class DexNet_cli(object):
    API = {0: ('Open a database', 'open_database'),
           1: ('Open a dataset', 'open_dataset'),
           2: ('Display object', 'display_object'),
           3: ('Display stable poses for object', 'display_stable_poses'),
           4: ('Display grasps for object', 'display_grasps'),
           5: ('Generate simulation data for object', 'compute_simulation_data'),
           6: ('Compute metadata', 'compute_metadata'),
           7: ('Display metadata', 'display_metadata'),
           8: ('Export objects', 'export_objects'),
           9: ('Set config (advanced)', 'set_config'),
           10: ('Quit', 'close')
           }

    def __init__(self):
        # init core members
        self.dexnet_api = dexnet.DexNet()

        # setup command line parsing
        self.comp = Completer()
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self.comp.complete)

        # display welcome message
        self.display_welcome()

    def display_welcome(self):
        print('####################################################')
        print('DEX-NET 0.1 Command Line Interface')
        print('Brought to you by AutoLab, UC Berkeley')
        print('####################################################')
        print()

    def display_menu(self):
        print()
        print('AVAILABLE COMMANDS:')
        for command_id, command_desc in DexNet_cli.API.items():
            print('%d) %s' % (command_id, command_desc[0]))
        print()

    def run_user_command(self):
        command = input('Enter a numeric command: ')
        try:
            try:
                command_id = int(command)
                if command_id not in DexNet_cli.API.keys():
                    raise RuntimeError()
                command_fn = getattr(self, DexNet_cli.API[command_id][1])
            except:
                raise RuntimeError()
            self.comp.set_words([])
            return command_fn()
        except RuntimeError:
            print('Command %s not recognized, please try again' % (command))
        except Exception as e:
            print('Error occurred!')
            self.close()
            raise
        return True

    def _get_checked_input(self, validity_check_fn, what_to_get):
        invalid_input = True
        while invalid_input:
            in_name = input('Enter {}: '.format(what_to_get))
            tokens = in_name.split()
            if len(tokens) > 1:
                print('Please provide only a single input')
            if in_name.lower() == 'q':
                return None
            invalid_input = not validity_check_fn(in_name)
            if invalid_input:
                print('Invalid input')
        return in_name

    def _get_fixed_input(self, valid_inputs, what_to_get):
        print()
        print('Available values:')
        for key in valid_inputs:
            if key is not '': print(key)
        print()
        self.comp.set_words(valid_inputs)
        return self._get_checked_input(lambda x: x in valid_inputs, what_to_get)

    def _get_yn_input(self, question):
        yn = input('{} [y/n] '.format(question))
        while yn.lower() != 'n' and yn.lower() != 'y':
            print('Did not understand input. Please answer \'y\' or \'n\'')
            yn = input('{} [y/n] '.format(question))
        return yn.lower() == 'y'

    def _check_opens(self):
        """ Checks that database and dataset are open """
        try:
            self.dexnet_api._check_opens()
            return True
        except RuntimeError as e:
            print(str(e))
            return False

    # commands below
    def open_database(self):
        """ Open a database """
        self.dexnet_api.close_database()

        database_name = self._get_checked_input(lambda x: True, "database name")
        if database_name is None: return True
        if os.path.splitext(database_name)[1] == '':
            database_name += dexnet.HDF5_EXT
        if not os.path.exists(database_name) and not os.path.exists(database_name):
            print('Database {} does not exist'.format(database_name))
            return True
        try:
            self.dexnet_api.open_database(database_name, create_db=True)
            print('Opened database %s' % (database_name))
            print()
            existing_datasets = [d.name for d in self.dexnet_api.database.datasets]
            if len(existing_datasets) == 1:
                dataset_name = existing_datasets[0]
                self.dexnet_api.open_dataset(dataset_name)
                print('Database has one dataset, opened dataset {}'.format(dataset_name))
        except Exception as e:
            print("Opening database failed: {}".format(str(e)))
        return True

    def open_dataset(self):
        """ Open a dataset """
        if self.dexnet_api.database is None:
            print('You must open a database first')
            return True

        # show existing datasets
        existing_datasets = [d.name for d in self.dexnet_api.database.datasets]
        print('Existing datasets:')
        for dataset_name in existing_datasets:
            print(dataset_name)
        print()
        # dexnet entity tab completion
        self.comp.set_words(existing_datasets)

        dataset_name = self._get_checked_input(lambda x: True, "dataset name")
        if dataset_name is None: return True
        if dataset_name not in existing_datasets:
            print('Dataset {} does not exist'.format(dataset_name))
            return True
        try:
            self.dexnet_api.open_dataset(dataset_name)
            print('Opened dataset {}'.format(dataset_name))
            print
        except Exception as e:
            print("Opening dataset failed: {}".format(str(e)))
        return True

    def compute_simulation_data(self):
        """ Preprocesses an object for simulation. """
        if not self._check_opens(): return True
        objects = self.dexnet_api.list_objects()
        obj_name = self._get_fixed_input(objects + [''], "object key [ENTER for entire dataset]")
        if obj_name is None: return True

        if obj_name == '':
            obj_names = objects
        else:
            obj_names = [obj_name]
        for obj_name in obj_names:
            try:
                self.dexnet_api.compute_simulation_data(obj_name)
            except Exception as e:
                print("Computing simulation preprocessing failed: {}".format(str(e)))
        return True

    def compute_metadata(self):
        """ Compute metadata for an object or the entire dataset """
        if not self._check_opens(): return True
        objects = self.dexnet_api.list_objects()
        obj_name = self._get_fixed_input(objects + [''], "object key [ENTER for entire dataset]")
        if obj_name is None: return True

        if obj_name == '':
            obj_names = objects
        else:
            obj_names = [obj_name]
        for obj_name in obj_names:
            try:
                self.dexnet_api.compute_metadata(obj_name)
            except Exception as e:
                print("Computing metadata failed: {}".format(str(e)))
        return True

    def display_metadata(self):
        """ View metadata for one object """
        if not self._check_opens(): return True
        objects = self.dexnet_api.list_objects()
        obj_name = self._get_fixed_input(objects, "object key")
        if obj_name is None: return True

        try:
            metadata = self.dexnet_api.dataset.object_metadata(obj_name)
            if len(metadata.keys()) == 0:
                print("No metadata available for object {}".format(obj_name))
            else:
                for key, val in metadata.iteritems():
                    print("{} : {}".format(key, val))
                print()
        except Exception as e:
            print("Display metadata failed: {}".format(str(e)))
        return True

    # TODO
    # Make this more general (support more metadata types)
    def export_objects(self):
        """ Export objects (filtered) """
        if not self._check_opens(): return True
        export_path = self._get_checked_input(lambda x: True, "path to directory to export to")
        if export_path is None: return True
        con_comp_filter = self._get_checked_input(lambda x: x.isdigit() or x == '',
                                                  "Max connected components (positive int) [ENTER for no filter]")
        if con_comp_filter is None: return True
        watertight_filter = self._get_yn_input("Restrict to watertight meshes?")
        filter_dict = {}
        if con_comp_filter != '':
            filter_dict['num_con_comps'] = lambda x: x <= con_comp_filter
        if watertight_filter:
            filter_dict['watertightness'] = lambda x: x

        self.dexnet_api.export_objects(export_path, filter_dict)
        return True

    def display_object(self):
        """ Display an object """
        if not self._check_opens(): return True
        objects = self.dexnet_api.list_objects()
        object_name = self._get_fixed_input(objects, "object key")
        if object_name is None: return True

        try:
            self.dexnet_api.display_object(object_name)
        except Exception as e:
            print("Display object failed: {}".format(str(e)))
        return True

    def display_stable_poses(self):
        """ Display stable poses """
        if not self._check_opens(): return True
        objects = self.dexnet_api.list_objects()
        object_name = self._get_fixed_input(objects, "object key")
        if object_name is None: return True

        try:
            self.dexnet_api.display_stable_poses(object_name)
        except Exception as e:
            print("Display object failed: {}".format(str(e)))
        return True

    def display_grasps(self):
        """ Display grasps for an object """
        if not self._check_opens(): return True
        grippers = self.dexnet_api.list_grippers()
        gripper_name = self._get_fixed_input(grippers, "gripper name")
        if gripper_name is None: return True
        objects = self.dexnet_api.list_objects()
        object_name = self._get_fixed_input(objects, "object key")
        if object_name is None: return True
        metrics = self.dexnet_api.list_metrics()
        metric_name = self._get_fixed_input(metrics, "metric name")
        if metric_name is None: return True

        try:
            self.dexnet_api.display_grasps(object_name, gripper_name, metric_name)
        except Exception as e:
            print("Display grasps failed: {}".format(str(e)))
        return True

    def set_config(self):
        """ Set fields in default config """
        config_dict = self.dexnet_api.default_config.config

        value_set = False
        while not value_set:
            fields = config_dict.keys()
            field_name = self._get_fixed_input(fields, "field name")
            if field_name is None: return True

            if isinstance(config_dict[field_name], collections.Mapping):
                print()
                print("Field {} has components:".format(field_name))
                config_dict = config_dict[field_name]
            else:
                old_val = config_dict[field_name]
                print()
                print("Field {} currently has value {}".format(field_name, old_val))
                new_val = self._get_checked_input(lambda x: True, "new value")
                if new_val is None: return True
                try:
                    new_val = type(old_val)(new_val)
                except:
                    new_type = 'str'
                    try:
                        new_val = float(new_val)
                        new_type = 'float'
                    except:
                        pass

                    print("New value does not match type of old value")
                    print("Old value had type {}, new value has type {}".format(type(old_val).__name__, new_type))
                    ok = self._get_yn_input("Is this OK?")
                    if not ok:
                        print("Setting parameter aborted")
                        return True
                config_dict[field_name] = new_val
                value_set = True
        return True

    def close(self):
        print('Closing Dex-Net. Goodbye!')
        self.dexnet_api.close_database()
        return False


if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Command line interface to Dex-Net')
    args = parser.parse_args()

    # open dex-net handle
    dexnet_cli = DexNet_cli()

    # setup graceful exit
    continue_dexnet = True


    def close_dexnet(signal=0, frame=None):
        dexnet_cli.close()
        continue_dexnet = False
        exit(0)


    signal.signal(signal.SIGINT, close_dexnet)

    # main loop
    while continue_dexnet:
        # display menu
        dexnet_cli.display_menu()

        # get user input
        continue_dexnet = dexnet_cli.run_user_command()
