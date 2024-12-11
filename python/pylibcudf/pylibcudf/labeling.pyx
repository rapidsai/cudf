# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport labeling as cpp_labeling
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.labeling cimport inclusive

from pylibcudf.libcudf.labeling import inclusive as Inclusive  # no-cython-lint

from .column cimport Column

__all__ = ["Inclusive", "label_bins"]

cpdef Column label_bins(
    Column input,
    Column left_edges,
    inclusive left_inclusive,
    Column right_edges,
    inclusive right_inclusive
):
    """Labels elements based on membership in the specified bins.

    For details see :cpp:func:`label_bins`.

    Parameters
    ----------
    input : Column
        Column of input elements to label according to the specified bins.
    left_edges : Column
        Column of the left edge of each bin.
    left_inclusive : Inclusive
        Whether or not the left edge is inclusive.
    right_edges : Column
        Column of the right edge of each bin.
    right_inclusive : Inclusive
        Whether or not the right edge is inclusive.

    Returns
    -------
    Column
        Column of integer labels of the elements in `input`
        according to the specified bins.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_labeling.label_bins(
            input.view(),
            left_edges.view(),
            left_inclusive,
            right_edges.view(),
            right_inclusive,
        )

    return Column.from_libcudf(move(c_result))
