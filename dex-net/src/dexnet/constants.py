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
# Grasp contact params
NO_CONTACT_DIST = 0.2  # distance to points that are not in contact for window extraction
WIN_DIST_LIM = 0.02  # limits for window plotting

# File extensions
HDF5_EXT = '.hdf5'
OBJ_EXT = '.obj'
OFF_EXT = '.off'
STL_EXT = '.stl'
SDF_EXT = '.sdf'
URDF_EXT = '.urdf'

# Tags for intermediate files
DEC_TAG = '_dec'
PROC_TAG = '_proc'

# Solver default max iterations
DEF_MAX_ITER = 100

# Access levels for db
READ_ONLY_ACCESS = 'READ_ONLY'
READ_WRITE_ACCESS = 'READ_WRITE'
WRITE_ACCESS = 'WRITE'
