# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum
from libcpp.string cimport string
from cudf._libxx.cpp.utilities.nvtx_utils cimport (
    range_push as cpp_range_push,
    range_push_hex as cpp_range_push_hex,
    range_pop as cpp_range_pop,
    color as color_types,
    underlying_type_t_color,
)


class Color(IntEnum):
    GREEN = <underlying_type_t_color> color_types.GREEN
    BLUE = <underlying_type_t_color> color_types.BLUE
    YELLOW = <underlying_type_t_color> color_types.YELLOW
    PURPLE = <underlying_type_t_color> color_types.PURPLE
    CYAN = <underlying_type_t_color> color_types.CYAN
    RED = <underlying_type_t_color> color_types.RED
    WHITE = <underlying_type_t_color> color_types.WHITE
    DARK_GREEN = <underlying_type_t_color> color_types.DARK_GREEN
    ORANGE = <underlying_type_t_color> color_types.ORANGE


def range_push(object name, object color='GREEN'):
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
    try:
        color = int(color, 16)
    except ValueError:
        color = int(Color[color.upper()].value)

    cdef const char *_name
    name = name.encode()
    _name = name

    cdef underlying_type_t_color _color = color

    with nogil:
        cpp_range_push_hex(_name, _color)


def range_pop():
    """
    Demarcate the end of a user-defined NVTX range.
    """
    with nogil:
        cpp_range_pop()
