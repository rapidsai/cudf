# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.nvtx cimport (
    range_push as cpp_range_push,
    range_push_hex as cpp_range_push_hex,
    range_pop as cpp_range_pop,

    _color as nvtx_color,
)


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
        result = cpp_range_push_hex(name, color)
    except ValueError:
        if color == 'green':
            result = cpp_range_push(name, nvtx_color.GREEN)
        elif color == 'blue':
            result = cpp_range_push(name, nvtx_color.BLUE)
        elif color == 'yellow':
            result = cpp_range_push(name, nvtx_color.YELLOW)
        elif color == 'purple':
            result = cpp_range_push(name, nvtx_color.PURPLE)
        elif color == 'cyan':
            result = cpp_range_push(name, nvtx_color.CYAN)
        elif color == 'red':
            result = cpp_range_push(name, nvtx_color.RED)
        elif color == 'white':
            result = cpp_range_push(name, nvtx_color.WHITE)
        elif color == 'darkgreen':
            result = cpp_range_push(name, nvtx_color.DARK_GREEN)
        elif color == 'orange':
            result = cpp_range_push(name, nvtx_color.ORANGE)
    print(result)


def range_pop():
    """
    Demarcate the end of a user-defined NVTX range.
    """
    result = cpp_range_pop()
    print(result)
