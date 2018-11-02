"""
Render modes
Author: Jeff Mahler
"""
class RenderMode(object):
    """Supported rendering modes.
    """
    SEGMASK = 'segmask'
    DEPTH = 'depth'
    DEPTH_SCENE = 'depth_scene'
    SCALED_DEPTH = 'scaled_depth'
    COLOR = 'color'
    COLOR_SCENE = 'color_scene'
    GRAY = 'gray'
    GD = 'gd'
    RGBD = 'rgbd'
    RGBD_SCENE = 'rgbd_scene'
    GRAYSCALE = 'gray'
