# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum
from libc.stdint cimport uint32_t

from cudf._libxx.cpp.nvtx cimport (
    range_push as cpp_range_push,
    range_push_hex as cpp_range_push_hex,
    range_pop as cpp_range_pop,

    color_type,
    underlying_type_t_color

)

class Color(IntEnum):
    GREEN = <underlying_type_t_color> color_type.GREEN
    BLUE = <underlying_type_t_color> color_type.BLUE
    YELLOW = <underlying_type_t_color> color_type.YELLOW
    PURPLE = <underlying_type_t_color> color_type.PURPLE
    CYAN = <underlying_type_t_color> color_type.CYAN
    RED = <underlying_type_t_color> color_type.RED
    WHITE = <underlying_type_t_color> color_type.WHITE
    DARK_GREEN = <underlying_type_t_color> color_type.DARK_GREEN
    ORANGE = <underlying_type_t_color> color_type.ORANGE

def range_push(object name, object color='green'):
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
    cdef const char* _name = name
    cdef uint32_t _color = color
    cdef color_type _Color = Color[color]
    try:
        _color = int(_color, 16)
        with nogil:
            cpp_range_push_hex(_name, _color)
    except ValueError:
        with nogil:
            cpp_range_push(_name, _Color)


def range_pop():
    """
    Demarcate the end of a user-defined NVTX range.
    """
    with nogil:
        cpp_range_pop()
