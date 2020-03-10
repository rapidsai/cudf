# Copyright (c) 2019, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t

ctypedef uint32_t underlying_type_t_color

cdef extern from "cudf/utilities/nvtx_utils.hpp" namespace "cudf::nvtx" nogil:
    ctypedef enum color_type 'color':
        GREEN 'cudf::nvtx::color::GREEN'
        BLUE 'cudf::nvtx::color::BLUE'
        YELLOW 'cudf::nvtx::color::YELLOW'
        PURPLE 'cudf::nvtx::color::PURPLE'
        CYAN 'cudf::nvtx::color::CYAN'
        RED 'cudf::nvtx::color::RED'
        WHITE 'cudf::nvtx::color::WHITE'
        DARK_GREEN 'cudf::nvtx::color::DARK_GREEN'
        ORANGE 'cudf::nvtx::color::ORANGE'

    color_type JOIN_COLOR 'cudf::nvtx::JOIN_COLOR'
    color_type GROUP_COLOR 'cudf::nvtx::GROUP_COLOR'
    color_type BINARY_OP_COLOR 'cudf::nvtx::BINARY_OP_COLOR'
    color_type PARTITION_COLOR 'cudf::nvtx::PARTITION_COLOR'
    color_type READ_CSV_COLOR 'cudf::nvtx::READ_CSV_COLOR'

    cdef void range_push(
        const char* const name, color_type color
    ) except +

    cdef void range_push_hex(
        const char* const name,
        uint32_t color
    ) except +

    cdef void range_pop() except +
