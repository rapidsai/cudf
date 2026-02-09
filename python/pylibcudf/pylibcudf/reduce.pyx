# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.functional cimport reference_wrapper
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional, nullopt
from libcpp.utility cimport move, pair
from pylibcudf.libcudf.aggregation cimport reduce_aggregation, scan_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.reduce cimport (
    reduce as cpp_reduce,
    scan as cpp_scan,
    minmax as cpp_minmax,
    scan_type,
    constscalar,
    is_valid_aggregation as cpp_is_valid_aggregation,
)
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport null_policy
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource

from pylibcudf.libcudf.reduce import scan_type as ScanType  # no-cython-lint

__all__ = ["ScanType", "minmax", "reduce", "scan", "is_valid_reduce_aggregation"]

cpdef Scalar reduce(
    Column col,
    Aggregation agg,
    DataType data_type,
    Scalar init=None,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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
    init : Scalar | None
        The initial value for the reduction.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned scalar's device memory.

    Returns
    -------
    Scalar
        The result of the reduction.
    """
    cdef unique_ptr[scalar] result
    cdef const reduce_aggregation *c_agg = agg.view_underlying_as_reduce()
    cdef optional[reference_wrapper[constscalar]] c_init
    cdef const scalar* c_init_ptr

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if init is not None:
        c_init_ptr = init.get()
        c_init = optional[reference_wrapper[constscalar]](
            reference_wrapper[constscalar](dereference(c_init_ptr))
        )
    else:
        c_init = nullopt

    with nogil:
        result = cpp_reduce(
            col.view(),
            dereference(c_agg),
            data_type.c_obj,
            c_init,
            stream.view(),
            mr.get_mr()
        )
    return Scalar.from_libcudf(move(result))


cpdef Column scan(
    Column col,
    Aggregation agg,
    scan_type inclusive,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
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
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The result of the scan.
    """
    cdef unique_ptr[column] result
    cdef const scan_aggregation *c_agg = agg.view_underlying_as_scan()

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_scan(
            col.view(),
            dereference(c_agg),
            inclusive,
            null_policy.EXCLUDE,
            stream.view(),
            mr.get_mr()
        )
    return Column.from_libcudf(move(result), stream, mr)


cpdef tuple minmax(Column col, Stream stream=None, DeviceMemoryResource mr=None):
    """Compute the minimum and maximum of a column

    For details, see ``cudf::minmax`` documentation.

    Parameters
    ----------
    col : Column
        The column to compute the minimum and maximum of.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned scalars' device memory.

    Returns
    -------
    tuple
        A tuple of two Scalars, the first being the minimum and the second
        being the maximum.
    """
    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] result
    cdef Scalar min_scalar
    cdef Scalar max_scalar

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_minmax(col.view(), stream.view(), mr.get_mr())

    min_scalar = Scalar.from_libcudf(move(result.first))
    max_scalar = Scalar.from_libcudf(move(result.second))
    return (min_scalar, max_scalar)


cpdef bool is_valid_reduce_aggregation(DataType source, Aggregation agg):
    """
    Return if an aggregation is supported for a given datatype.

    Parameters
    ----------
    source
        The type of the column the aggregation is being performed on.
    agg
        The aggregation.

    Returns
    -------
    True if the aggregation is supported.
    """
    return cpp_is_valid_aggregation(source.c_obj, agg.kind())


ScanType.__str__ = ScanType.__repr__
