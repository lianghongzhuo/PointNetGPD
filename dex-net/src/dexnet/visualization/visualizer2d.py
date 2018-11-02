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
Dex-Net 2D visualizer extension
Author: Jeff Mahler
"""
import copy
import json
import IPython
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.spatial.distance as ssd

from visualization import Visualizer2D


class DexNetVisualizer2D(Visualizer2D):
    """
    Dex-Net extension of the base pyplot 2D visualization tools
    """

    @staticmethod
    def grasp(grasp, color='r', arrow_len=4, arrow_head_len=2, arrow_head_width=3,
              arrow_width=1, jaw_len=3, jaw_width=3.0,
              grasp_center_size=7.5, grasp_center_thickness=2.5,
              grasp_center_style='+', grasp_axis_width=1,
              grasp_axis_style='--', line_width=8.0, show_center=True, show_axis=False, scale=1.0):
        """
        Plots a 2D grasp with arrow and jaw style using matplotlib
        
        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            2D grasp to plot
        color : :obj:`str`
            color of plotted grasp
        arrow_len : float
            length of arrow body
        arrow_head_len : float
            length of arrow head
        arrow_head_width : float
            width of arrow head
        arrow_width : float
            width of arrow body
        jaw_len : float
            length of jaw line
        jaw_width : float
            line width of jaw line
        grasp_center_thickness : float
            thickness of grasp center
        grasp_center_style : :obj:`str`
            style of center of grasp
        grasp_axis_width : float
            line width of grasp axis
        grasp_axis_style : :obj:`str
            style of grasp axis line
        show_center : bool
            whether or not to plot the grasp center
        show_axis : bool
            whether or not to plot the grasp axis
        """
        # plot grasp center
        if show_center:
            plt.plot(grasp.center[1], grasp.center[0], c=color, marker=grasp_center_style,
                     mew=scale * grasp_center_thickness, ms=scale * grasp_center_size)

        # compute axis and jaw locations
        axis = np.array([np.sin(grasp.angle), np.cos(grasp.angle)])
        g1 = grasp.center - (grasp.width / 2) * axis
        g2 = grasp.center + (grasp.width / 2) * axis
        g1p = g1 - scale * arrow_len * axis  # start location of grasp jaw 1
        g2p = g2 + scale * arrow_len * axis  # start location of grasp jaw 2

        # plot grasp axis
        if show_axis:
            plt.plot([g1[1], g2[1]], [g1[0], g2[0]], color=color, linewidth=scale * grasp_axis_width,
                     linestyle=grasp_axis_style)

        # direction of jaw line
        jaw_dir = scale * jaw_len * np.array([axis[1], -axis[0]])

        # length of arrow
        alpha = scale * (arrow_len - arrow_head_len)

        # plot first jaw
        g1_line = np.c_[g1p, g1 - scale * arrow_head_len * axis].T
        # plt.plot(g1_line[:,1], g1_line[:,0], linewidth=scale*line_width, c=color)
        plt.arrow(g1p[1], g1p[0], alpha * axis[1], alpha * axis[0], width=scale * arrow_width,
                  head_width=scale * arrow_head_width, head_length=scale * arrow_head_len, fc=color, ec=color)
        jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T
        plt.plot(jaw_line1[:, 1], jaw_line1[:, 0], linewidth=scale * jaw_width, c=color)

        # plot second jaw
        g2_line = np.c_[g2p, g2 + scale * arrow_head_len * axis].T
        # plt.plot(g2_line[:,1], g2_line[:,0], linewidth=scale*line_width, c=color)
        plt.arrow(g2p[1], g2p[0], -alpha * axis[1], -alpha * axis[0], width=scale * arrow_width,
                  head_width=scale * arrow_head_width, head_length=scale * arrow_head_len, fc=color, ec=color)
        jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
        plt.plot(jaw_line2[:, 1], jaw_line2[:, 0], linewidth=scale * jaw_width, c=color)
