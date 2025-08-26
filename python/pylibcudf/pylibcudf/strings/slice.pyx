# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_fixed_width_scalar as cpp_make_fixed_width_scalar,
)
from pylibcudf.libcudf.strings cimport substring as cpp_slice
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

from cython.operator import dereference
from rmm.pylibrmm.stream cimport Stream

from ..utils cimport _get_stream

__all__ = ["slice_strings"]

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=None,
    ColumnOrScalar stop=None,
    Scalar step=None,
    Stream stream=None
):
    """Perform a slice operation on a strings column.

    ``start`` and ``stop`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`. But ``step`` must be a
    :py:class:`~pylibcudf.scalar.Scalar`.

    For details, see :cpp:func:`cudf::strings::slice_strings`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    start : Union[Column, Scalar]
        The start character position or positions.
    stop : Union[Column, Scalar]
        The end character position or positions
    step : Scalar
        Distance between input characters retrieved
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        The result of the slice operation
    """
    cdef unique_ptr[column] c_result
    cdef numeric_scalar[size_type]* cpp_start
    cdef numeric_scalar[size_type]* cpp_stop
    cdef numeric_scalar[size_type]* cpp_step
    stream = _get_stream(stream)

    if input is None:
        raise ValueError("input cannot be None")

    if ColumnOrScalar is Column:
        if step is not None:
            raise ValueError("Column-wise slice does not support step")

        if start is None or stop is None:
            raise ValueError(
                "start and stop must be provided for Column-wise slice"
            )

        with nogil:
            c_result = cpp_slice.slice_strings(
                input.view(),
                start.view(),
                stop.view(),
                stream.view()
            )

    elif ColumnOrScalar is Scalar:
        if start is None:
            stream = _get_stream(None)
            start = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(0, stream.view())
            )
        if stop is None:
            stream = _get_stream(None)
            stop = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(0, stream.view())
            )
        if step is None:
            stream = _get_stream(None)
            step = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(1, stream.view())
            )

        cpp_start = <numeric_scalar[size_type]*>start.c_obj.get()
        cpp_stop = <numeric_scalar[size_type]*>stop.c_obj.get()
        cpp_step = <numeric_scalar[size_type]*>step.c_obj.get()

        with nogil:
            c_result = cpp_slice.slice_strings(
                input.view(),
                dereference(cpp_start),
                dereference(cpp_stop),
                dereference(cpp_step),
                stream.view()
            )
    else:
        raise ValueError("start, stop, and step must be either Column or Scalar")

    return Column.from_libcudf(move(c_result), stream)
