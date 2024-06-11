# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from cudf._lib.pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_fixed_width_scalar as cpp_make_fixed_width_scalar,
)
from cudf._lib.pylibcudf.libcudf.strings cimport substring as cpp_slice
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.scalar cimport Scalar

from cython.operator import dereference


cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=None,
    ColumnOrScalar stop=None,
    ColumnOrScalar step=None
):
    cdef unique_ptr[column] c_result
    cdef numeric_scalar[size_type]* cpp_start
    cdef numeric_scalar[size_type]* cpp_stop
    cdef numeric_scalar[size_type]* cpp_step

    if ColumnOrScalar is Column:
        pass
    elif ColumnOrScalar is Scalar:
        if start is None:
            delimiters = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(0)
            )
        if stop is None:
            delimiters = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(0)
            )
        if step is None:
            delimiters = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(1)
            )

        cpp_start = <numeric_scalar[size_type]*>start.c_obj.get()
        cpp_stop = <numeric_scalar[size_type]*>stop.c_obj.get()
        cpp_step = <numeric_scalar[size_type]*>step.c_obj.get()

        with nogil:
            c_result = cpp_slice.slice_strings(
                input.view(),
                dereference(cpp_start),
                dereference(cpp_stop),
                dereference(cpp_step)
            )
    else:
        raise ValueError("start, stop, and step must be either Column or Scalar")

    return Column.from_libcudf(move(c_result))
