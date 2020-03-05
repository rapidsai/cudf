# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t

from cudf._libxx.cpp.nvtx cimport (
    range_push as cpp_range_push,
    range_push_hex as cpp_range_push_hex,
    range_pop as cpp_range_pop,

    _color as nvtx_color,
)


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
    try:
        _color = int(_color, 16)
        with nogil:
            cpp_range_push_hex(_name, _color)
    except ValueError:
        if color == 'green':
            with nogil:
                cpp_range_push(_name, nvtx_color.GREEN)
        elif color == 'blue':
            with nogil:
                cpp_range_push(_name, nvtx_color.BLUE)
        elif color == 'yellow':
            with nogil:
                cpp_range_push(_name, nvtx_color.YELLOW)
        elif color == 'purple':
            with nogil:
                cpp_range_push(_name, nvtx_color.PURPLE)
        elif color == 'cyan':
            with nogil:
                cpp_range_push(_name, nvtx_color.CYAN)
        elif color == 'red':
            with nogil:
                cpp_range_push(_name, nvtx_color.RED)
        elif color == 'white':
            with nogil:
                cpp_range_push(_name, nvtx_color.WHITE)
        elif color == 'darkgreen':
            with nogil:
                cpp_range_push(_name, nvtx_color.DARK_GREEN)
        elif color == 'orange':
            with nogil:
                cpp_range_push(_name, nvtx_color.ORANGE)



def range_pop():
    """
    Demarcate the end of a user-defined NVTX range.
    """
    with nogil:
        cpp_range_pop()