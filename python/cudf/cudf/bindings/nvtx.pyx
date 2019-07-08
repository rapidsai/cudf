# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *


_GDF_COLORS = {
    'green':    GDF_GREEN,
    'blue':     GDF_BLUE,
    'yellow':   GDF_YELLOW,
    'purple':   GDF_PURPLE,
    'cyan':     GDF_CYAN,
    'red':      GDF_RED,
    'white':    GDF_WHITE,
    'darkgreen': GDF_DARK_GREEN,
    'orange':   GDF_ORANGE,
}


def str_to_gdf_color(s):
    """Util to convert str to gdf_color type.
    """
    return _GDF_COLORS[s.lower()]


def nvtx_range_push(name, color='green'):
    """
    Demarcate the beginning of a user-defined NVTX range.

    Parameters
    ----------
    name : str
        The name of the NVTX range
    color : str
        The color to use for the range.
        Can be named color or hex RGB string.
    """
    name = name.encode('ascii')
    try:
        color = int(color, 16)
        result = gdf_nvtx_range_push_hex(name, color)
    except ValueError:
        color = _GDF_COLORS[color]
        result = gdf_nvtx_range_push(name, color)
    check_gdf_error(result)


def nvtx_range_pop():
    """
    Demarcate the end of a user-defined NVTX range.
    """
    result = gdf_nvtx_range_pop()
    check_gdf_error(result)
