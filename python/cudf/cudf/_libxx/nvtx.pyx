# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from cudf._libxx.lib import *

def range_push(name, color='green'):
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
        result = range_push_hex(name, color)
    except ValueError:
        result = range_push(name, color)
    check_gdf_error(result)


def range_pop():
    """
    Demarcate the end of a user-defined NVTX range.
    """
    result = range_pop()
    check_gdf_error(result)
