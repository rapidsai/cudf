# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair
from pylibcudf.libcudf cimport reduce as cpp_reduce
from pylibcudf.libcudf.aggregation cimport reduce_aggregation, scan_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.reduce cimport scan_type
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport null_policy
from rmm.pylibrmm.stream cimport Stream

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType
from .utils cimport _get_stream

from pylibcudf.libcudf.reduce import scan_type as ScanType  # no-cython-lint

__all__ = ["ScanType", "minmax", "reduce", "scan"]

cpdef Scalar reduce(
    Column col,
    Aggregation agg,
    DataType data_type,
    Stream stream=None
):
    """Perform a reduction on a column

    For details, see ``cudf::reduce`` documentation.

    Parameters
    ----------
    col : Column
        The column to perform the reduction on.
    agg : Aggregation
        The aggregation to perform.
    data_type : DataType
        The data type of the result.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Scalar
        The result of the reduction.
    """
    cdef unique_ptr[scalar] result
    cdef const reduce_aggregation *c_agg = agg.view_underlying_as_reduce()

    stream = _get_stream(stream)

    with nogil:
        result = cpp_reduce.cpp_reduce(
            col.view(),
            dereference(c_agg),
            data_type.c_obj,
            stream.view()
        )
    return Scalar.from_libcudf(move(result), stream)


cpdef Column scan(Column col, Aggregation agg, scan_type inclusive, Stream stream=None):
    """Perform a scan on a column

    For details, see ``cudf::scan`` documentation.

    Parameters
    ----------
    col : Column
        The column to perform the scan on.
    agg : Aggregation
        The aggregation to perform.
    inclusive : scan_type
        The type of scan to perform.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        The result of the scan.
    """
    cdef unique_ptr[column] result
    cdef const scan_aggregation *c_agg = agg.view_underlying_as_scan()

    stream = _get_stream(stream)

    with nogil:
        result = cpp_reduce.cpp_scan(
            col.view(),
            dereference(c_agg),
            inclusive,
            null_policy.EXCLUDE,
            stream.view(),
        )
    return Column.from_libcudf(move(result), stream)


cpdef tuple minmax(Column col, Stream stream=None):
    """Compute the minimum and maximum of a column

    For details, see ``cudf::minmax`` documentation.

    Parameters
    ----------
    col : Column
        The column to compute the minimum and maximum of.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    tuple
        A tuple of two Scalars, the first being the minimum and the second
        being the maximum.
    """
    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_reduce.cpp_minmax(col.view(), stream.view())

    return (
        Scalar.from_libcudf(move(result.first), stream),
        Scalar.from_libcudf(move(result.second), stream),
    )

ScanType.__str__ = ScanType.__repr__
