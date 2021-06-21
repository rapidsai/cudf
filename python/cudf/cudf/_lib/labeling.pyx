# Copyright (c) 2021, NVIDIA CORPORATION.

import numpy as np
from enum import IntEnum

from libc.stdint cimport uint32_t
from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.replace import replace_nulls

from cudf._lib.cpp.labeling cimport inclusive
from cudf._lib.cpp.labeling cimport label_bins as cpp_label_bins
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


# Note that the parameter input shadows a Python built-in in the local scope,
# but I'm not too concerned about that since there's no use-case for actual
# input in this context.
def label_bins(Column input, Column left_edges, cbool left_inclusive,
               Column right_edges, cbool right_inclusive):
    cdef inclusive c_left_inclusive = \
        inclusive.YES if left_inclusive else inclusive.NO
    cdef inclusive c_right_inclusive = \
        inclusive.YES if right_inclusive else inclusive.NO

    cdef column_view input_view = input.view()
    cdef column_view left_edges_view = left_edges.view()
    cdef column_view right_edges_view = right_edges.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_label_bins(
                input_view,
                left_edges_view,
                c_left_inclusive,
                right_edges_view,
                c_right_inclusive,
            )
        )

    return Column.from_unique_ptr(move(c_result))
