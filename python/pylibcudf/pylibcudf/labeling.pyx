# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport labeling as cpp_labeling
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.labeling cimport inclusive

from .column cimport Column


cpdef Column label_bins(
    Column input,
    Column left_edges,
    bool left_inclusive,
    Column right_edges,
    bool right_inclusive
):
    """Labels elements based on membership in the specified bins.

    Parameters
    ----------
    input : Column
        Column of input elements to label according to the specified bins.
    left_edges : Column
        Column of the left edge of each bin.
    left_inclusive : bool
        Whether or not the left edge is inclusive.
    right_edges : Column
        Column of the right edge of each bin.
    right_inclusive : bool
        Whether or not the right edge is inclusive.

    Returns
    -------
    Column
        Column of integer labels of the elements in `input`
        according to the specified bins.
    """
    cdef unique_ptr[column] c_result
    cdef inclusive c_left_inclusive = (
        inclusive.YES if left_inclusive else inclusive.NO
    )
    cdef inclusive c_right_inclusive = (
        inclusive.YES if right_inclusive else inclusive.NO
    )

    with nogil:
        c_result = move(
            cpp_labeling.label_bins(
                input.view(),
                left_edges.view(),
                c_left_inclusive,
                right_edges.view(),
                c_right_inclusive,
            )
        )

    return Column.from_libcudf(move(c_result))
