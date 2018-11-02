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
# Keys for easy lookups in HDF5 databases
METRICS_KEY = 'metrics'
OBJECTS_KEY = 'objects'
MESH_KEY = 'mesh'
SDF_KEY = 'sdf'
GRASPS_KEY = 'grasps'
GRIPPERS_KEY = 'grippers'
NUM_GRASPS_KEY = 'num_grasps'
LOCAL_FEATURES_KEY = 'local_features'
GLOBAL_FEATURES_KEY = 'global_features'
SHOT_FEATURES_KEY = 'shot'
RENDERED_IMAGES_KEY = 'rendered_images'
SENSOR_DATA_KEY = 'sensor_data'
STP_KEY = 'stable_poses'
CATEGORY_KEY = 'category'
MASS_KEY = 'mass'
CONVEX_PIECES_KEY = 'convex_pieces'

CREATION_KEY = 'time_created'
DATASETS_KEY = 'datasets'
DATASET_KEY = 'dataset'

# data keys for easy access
SDF_DATA_KEY = 'data'
SDF_ORIGIN_KEY = 'origin'
SDF_RES_KEY = 'resolution'
SDF_POSE_KEY = 'pose'
SDF_SCALE_KEY = 'scale'
SDF_FRAME_KEY = 'frame'

MESH_VERTICES_KEY = 'vertices'
MESH_TRIANGLES_KEY = 'triangles'
MESH_NORMALS_KEY = 'normals'
MESH_POSE_KEY = 'pose'
MESH_SCALE_KEY = 'scale'
MESH_DENSITY_KEY = 'density'

LOCAL_FEATURE_NUM_FEAT_KEY = 'num_features'
LOCAL_FEATURE_DESC_KEY = 'descriptors'
LOCAL_FEATURE_RF_KEY = 'rfs'
LOCAL_FEATURE_POINT_KEY = 'points'
LOCAL_FEATURE_NORMAL_KEY = 'normals'
SHOT_FEATURES_KEY = 'shot'
FEATURE_KEY = 'feature'

NUM_STP_KEY = 'num_stable_poses'
POSE_KEY = 'pose'
STABLE_POSE_PROB_KEY = 'p'
STABLE_POSE_ROT_KEY = 'r'
STABLE_POSE_PT_KEY = 'x0'

NUM_GRASPS_KEY = 'num_grasps'
GRASP_KEY = 'grasp'
GRASP_ID_KEY = 'id'
GRASP_TYPE_KEY = 'type'
GRASP_CONFIGURATION_KEY = 'configuration'
GRASP_RF_KEY = 'frame'
GRASP_TIMESTAMP_KEY = 'timestamp'
GRASP_METRICS_KEY = 'metrics'
GRASP_FEATURES_KEY = 'features'

GRASP_FEATURE_NAME_KEY = 'name'
GRASP_FEATURE_TYPE_KEY = 'type'
GRASP_FEATURE_VECTOR_KEY = 'vector'

NUM_IMAGES_KEY = 'num_images'
IMAGE_KEY = 'image'
IMAGE_DATA_KEY = 'image_data'
IMAGE_FRAME_KEY = 'image_frame'
CAM_POS_KEY = 'cam_pos'
CAM_ROT_KEY = 'cam_rot'
CAM_INT_PT_KEY = 'cam_int_pt'
CAM_FRAME_KEY = 'cam_frame'

# Extras
RENDERED_IMAGE_TYPES = ['segmask', 'depth', 'scaled_depth']

# Metadata
METADATA_KEY = 'metadata'
METADATA_TYPE_KEY = 'type'
METADATA_DESC_KEY = 'description'
METADATA_FUNC_KEY = 'func'

# Connected components
CONNECTED_COMPONENTS_KEY = 'connected_components'
