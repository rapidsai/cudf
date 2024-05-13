# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from cudf._lib.pylibcudf.libcudf cimport reduce as cpp_reduce
from cudf._lib.pylibcudf.libcudf.aggregation cimport (
    reduce_aggregation,
    scan_aggregation,
)
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.reduce cimport scan_type
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType

from cudf._lib.pylibcudf.libcudf.reduce import \
    scan_type as ScanType  # no-cython-lint


cpdef Scalar reduce(Column col, Aggregation agg, DataType data_type):
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

    Returns
    -------
    Scalar
        The result of the reduction.
    """
    cdef unique_ptr[scalar] result
    cdef const reduce_aggregation *c_agg = agg.view_underlying_as_reduce()
    with nogil:
        result = move(
            cpp_reduce.cpp_reduce(
                col.view(),
                dereference(c_agg),
                data_type.c_obj
            )
        )
    return Scalar.from_libcudf(move(result))


cpdef Column scan(Column col, Aggregation agg, scan_type inclusive):
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

    Returns
    -------
    Column
        The result of the scan.
    """
    cdef unique_ptr[column] result
    cdef const scan_aggregation *c_agg = agg.view_underlying_as_scan()
    with nogil:
        result = move(
            cpp_reduce.cpp_scan(
                col.view(),
                dereference(c_agg),
                inclusive,
            )
        )
    return Column.from_libcudf(move(result))


cpdef tuple minmax(Column col):
    """Compute the minimum and maximum of a column

    For details, see ``cudf::minmax`` documentation.

    Parameters
    ----------
    col : Column
        The column to compute the minimum and maximum of.

    Returns
    -------
    tuple
        A tuple of two Scalars, the first being the minimum and the second
        being the maximum.
    """
    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] result
    with nogil:
        result = move(cpp_reduce.cpp_minmax(col.view()))

    return (
        Scalar.from_libcudf(move(result.first)),
        Scalar.from_libcudf(move(result.second)),
    )
