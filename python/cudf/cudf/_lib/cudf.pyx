# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
from cudf._lib.cudf cimport *
from cudf._lib.GDFError import GDFError
from cudf._libxx.column cimport Column
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy
from rmm._lib.lib cimport c_free, cudaStream_t


import numpy as np
import pandas as pd
import pyarrow as pa


from cudf.utils import cudautils
from cudf.utils.dtypes import is_categorical_dtype
from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize
import rmm
import nvstrings
import nvcategory

dtypes = {
    np.float64: GDF_FLOAT64,
    np.float32: GDF_FLOAT32,
    np.int64: GDF_INT64,
    np.longlong: GDF_INT64,
    np.int32: GDF_INT32,
    np.int16: GDF_INT16,
    np.int8: GDF_INT8,
    np.bool_: GDF_BOOL8,
    np.datetime64: GDF_DATE64,
    np.object_: GDF_STRING_CATEGORY,
    np.str_: GDF_STRING_CATEGORY,
}

gdf_dtypes = {
    GDF_FLOAT64: np.float64,
    GDF_FLOAT32: np.float32,
    GDF_INT64: np.int64,
    GDF_INT32: np.int32,
    GDF_INT16: np.int16,
    GDF_INT8: np.int8,
    GDF_BOOL8: np.bool_,
    GDF_DATE64: np.datetime64,
    GDF_TIMESTAMP: np.datetime64,
    GDF_CATEGORY: np.int32,
    GDF_STRING_CATEGORY: np.object_,
    GDF_STRING: np.object_,
    N_GDF_TYPES: np.int32
}

agg_ops = {
    'sum': GDF_SUM,
    'max': GDF_MAX,
    'min': GDF_MIN,
    'mean': GDF_AVG,
    'avg': GDF_AVG,
    'count': GDF_COUNT,
    'count_distinct': GDF_COUNT_DISTINCT,
}

np_to_gdf_time_unit = {
    's': TIME_UNIT_s,
    'ms': TIME_UNIT_ms,
    'us': TIME_UNIT_us,
    'ns': TIME_UNIT_ns,
}

gdf_to_np_time_unit = {
    TIME_UNIT_s: 's',
    TIME_UNIT_ms: 'ms',
    TIME_UNIT_us: 'us',
    TIME_UNIT_ns: 'ns',
}

cpdef gdf_time_unit_to_np_dtype(gdf_time_unit time_unit):
    """Util to convert gdf_time_unit to numpy datetime64 dtype.
    """
    np_time_unit = gdf_to_np_time_unit.get(time_unit, 'ms')
    return np.dtype('datetime64[{}]'.format(np_time_unit))


cpdef gdf_time_unit np_dtype_to_gdf_time_unit(dtype):
    """Util to convert numpy datetime64 dtype to gdf_time_unit.
    """
    if dtype.kind != 'M':
        return TIME_UNIT_NONE
    np_time_unit, _ = np.datetime_data(dtype)
    return np_to_gdf_time_unit.get(np_time_unit, TIME_UNIT_NONE)


def check_gdf_compatibility(col):
    """
    Raise TypeError when a column type does not have gdf support.
    """
    if not (col.dtype.type in dtypes or is_categorical_dtype(col)):
        raise TypeError('column type `%s` not supported in gdf' % (col.dtype))


cpdef get_ctype_ptr(obj):
    if obj.device_ctypes_pointer.value is None:
        return 0
    else:
        return obj.device_ctypes_pointer.value


cpdef get_column_data_ptr(obj):
    return obj.data.ptr


cpdef get_column_valid_ptr(obj):
    return obj.mask.ptr


cdef np_dtype_from_gdf_column(gdf_column* col):
    """Util to convert a gdf_column's dtype to a numpy dtype.

    Parameters
    ----------
    col : gdf_column
        The gdf_column from which to infer a numpy.dtype.
    """
    dtype = col.dtype
    if dtype == GDF_DATE64:
        return np.dtype('datetime64[ms]')
    if dtype == GDF_TIMESTAMP:
        return gdf_time_unit_to_np_dtype(col.dtype_info.time_unit)
    if dtype in gdf_dtypes:
        return np.dtype(gdf_dtypes[dtype])
    raise TypeError('cannot convert gdf_dtype `%s` to numpy dtype' % (dtype))


cdef gdf_dtype gdf_dtype_from_dtype(dtype) except? GDF_invalid:
    # if dtype is pd.CategoricalDtype, use the codes' gdf_dtype
    if is_categorical_dtype(dtype):
        raise NotImplementedError()
    # if dtype is np.datetime64, interrogate the dtype's time_unit resolution
    if pd.api.types.is_datetime64_dtype(dtype):
        time_unit, _ = np.datetime_data(dtype)
        if time_unit in np_to_gdf_time_unit:
            # time_unit is valid so must be a GDF_TIMESTAMP
            return GDF_TIMESTAMP
        # else default to GDF_DATE64
        return GDF_DATE64
    # everything else is a 1-1 mapping
    dtype = np.dtype(dtype)
    if dtype.type in dtypes:
        return dtypes[dtype.type]
    raise TypeError('cannot convert numpy dtype `%s` to gdf_dtype' % (dtype))


cdef gdf_scalar* gdf_scalar_from_scalar(val, dtype=None) except? NULL:
    """
    Returns a gdf_scalar* constructed from the numpy scalar ``val``.
    """
    if (
        val is not None
        and not np.isscalar(val)
    ):
        raise TypeError("val must be a NumPy scalar or None")

    cdef bool is_valid = True

    if dtype is None:
        dtype = val.dtype

    cdef gdf_scalar* s = <gdf_scalar*>malloc(sizeof(gdf_scalar))
    if s is NULL:
        raise MemoryError
    if val is None:
        is_valid = False
        val = dtype.type(0)

    s[0].is_valid = is_valid
    s[0].dtype = gdf_dtype_from_dtype(dtype)

    set_scalar_value(s, val)

    return s


cdef get_scalar_value(gdf_scalar scalar, dtype):
    """
    Returns typed value from a gdf_scalar
    0-dim array is retuned if dtype is date32/64, timestamp
    """
    if scalar.dtype == GDF_FLOAT64:
        return scalar.data.fp64
    if scalar.dtype == GDF_FLOAT32:
        return scalar.data.fp32
    if scalar.dtype == GDF_INT64:
        return scalar.data.si64
    if scalar.dtype == GDF_INT32:
        return scalar.data.si32
    if scalar.dtype == GDF_INT16:
        return scalar.data.si16
    if scalar.dtype == GDF_INT8:
        return scalar.data.si08
    if scalar.dtype == GDF_BOOL8:
        return scalar.data.b08
    if scalar.dtype == GDF_DATE64:
        time_unit, _ = np.datetime_data(dtype)
        return dtype.type(scalar.data.dt64, time_unit)
    if scalar.dtype == GDF_TIMESTAMP:
        time_unit, _ = np.datetime_data(dtype)
        return dtype.type(scalar.data.tmst, time_unit)
    raise ValueError("Cannot convert gdf_scalar of dtype {}",
                     "to numpy scalar".format(scalar.dtype))


cdef set_scalar_value(gdf_scalar *scalar, val):
    """
    Sets the value of a gdf_scalar from a numpy scalar.
    """
    if val.dtype.type == np.float64:
        scalar.data.fp64 = val
    elif val.dtype.type == np.float32:
        scalar.data.fp32 = val
    elif val.dtype.type == np.int64 or val.dtype.type == np.longlong:
        scalar.data.si64 = val
    elif val.dtype.type == np.int32:
        scalar.data.si32 = val
    elif val.dtype.type == np.int16:
        scalar.data.si16 = val
    elif val.dtype.type == np.int8:
        scalar.data.si08 = val
    elif val.dtype.type == np.bool or val.dtype.type == np.bool_:
        scalar.data.b08 = val
    elif val.dtype.type == np.datetime64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in np_to_gdf_time_unit:
            scalar.data.tmst = np.int64(val)
        else:
            scalar.data.dt64 = np.int64(val)
    else:
        raise ValueError("Cannot convert numpy scalar of dtype {}"
                         "to gdf_scalar".format(val.dtype.name))


# gdf_column functions

cdef gdf_column* column_view_from_column(Column col,
                                         col_name=None) except? NULL:
    """
    Make a column view from a column

    Parameters
    ----------
    size: int
        Data count.
    data: Buffer
        The data buffer.
    mask: Buffer; optional
        The mask buffer.
    dtype: numpy.dtype; optional
        The dtype of the data.  Defaults to *data.dtype*.
    """
    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr
    cdef uintptr_t valid_ptr
    cdef uintptr_t category
    cdef gdf_dtype c_dtype

    if is_categorical_dtype(col.dtype):
        c_dtype = gdf_dtype_from_dtype(col.codes.dtype)
    else:
        c_dtype = gdf_dtype_from_dtype(col.dtype)

    if c_dtype == GDF_STRING_CATEGORY:
        category = col.nvcategory.get_cpointer()
        if len(col) > 0:
            data_ptr = get_ctype_ptr(col.indices)
        else:
            data_ptr = 0
    else:
        category = 0

        if len(col) > 0:
            if is_categorical_dtype(col.dtype):
                data_ptr = col.codes.data.ptr
            else:
                data_ptr = col.data.ptr
        else:
            data_ptr = 0

    if col.nullable:
        valid_ptr = col.mask.ptr
    else:
        valid_ptr = 0

    cdef char* c_col_name = py_to_c_str(col_name)
    cdef size_type len_col = len(col)
    cdef size_type c_null_count = col.null_count
    cdef gdf_time_unit c_time_unit = np_dtype_to_gdf_time_unit(col.dtype)
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(
        time_unit=c_time_unit,
        category=<void*>category
    )

    with nogil:
        gdf_column_view_augmented(
            <gdf_column*>c_col,
            <void*>data_ptr,
            <valid_type*>valid_ptr,
            len_col,
            c_dtype,
            c_null_count,
            c_extra_dtype_info,
            c_col_name
        )

    if hasattr(col, "null_count"):
        del col.null_count

    return c_col

cpdef uintptr_t column_view_pointer(col):
    """
    Return pointer to a view of the underlying <gdf_column*>
    """
    return <uintptr_t> column_view_from_column(col)


cdef Column gdf_column_to_column(gdf_column* c_col):
    """
    Util to create a Python cudf.Column from a libcudf gdf_column.

    Parameters
    ----------
    c_col : gdf_column*
        A pointer to the source gdf_column.
    """
    from cudf.core.buffer import Buffer
    from cudf.core.column import build_column, as_column
    from cudf.utils.utils import mask_bitsize, calc_chunk_size

    gdf_dtype = c_col.dtype
    data_ptr = int(<uintptr_t>c_col.data)

    if gdf_dtype == GDF_STRING:
        data = nvstrings.bind_cpointer(data_ptr)
        result = as_column(data)
    elif gdf_dtype == GDF_STRING_CATEGORY:
        c_free(<void*><uintptr_t>data_ptr, <cudaStream_t><uintptr_t>0)

        if c_col.size == 0:
            data = nvstrings.to_device([])
        else:
            nvcat_ptr = int(<uintptr_t>c_col.dtype_info.category)
            nvcat_obj = nvcategory.bind_cpointer(nvcat_ptr)
            data = nvcat_obj.to_strings()
        result = as_column(data)

    else:
        dtype = np_dtype_from_gdf_column(c_col)
        dptr = rmm._DevicePointer(data_ptr)
        data = Buffer(dptr, dtype.itemsize*c_col.size)

        mask = None
        if c_col.valid is not NULL:
            mptr = rmm._DevicePointer(int(<uintptr_t>c_col.valid))
            mask = Buffer(mptr, size=calc_chunk_size(c_col.size, mask_bitsize))

        result = build_column(data=data,
                              dtype=dtype,
                              mask=mask)

    return result


cdef gdf_column* column_view_from_string_column(
    Column col,
    col_name=None
) except? NULL:
    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr = col.nvstrings.get_cpointer()
    cdef uintptr_t mask_ptr
    cdef uintptr_t category = 0
    cdef gdf_dtype c_dtype = GDF_STRING

    if col.nullable and col.has_nulls:
        mask_ptr = col.mask.ptr
    else:
        mask_ptr = 0

    if col_name is None:
        col_name = col.name

    cdef char* c_col_name = py_to_c_str(col_name)
    cdef size_type len_col = len(col)
    cdef size_type c_null_count = col.null_count
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(
        time_unit=TIME_UNIT_NONE,
        category=<void*>category
    )

    with nogil:
        gdf_column_view_augmented(
            <gdf_column*>c_col,
            <void*>data_ptr,
            <valid_type*>mask_ptr,
            len_col,
            c_dtype,
            c_null_count,
            c_extra_dtype_info,
            c_col_name
        )

    if hasattr(col, "null_count"):
        del col.null_count

    return c_col


cdef gdf_column** cols_view_from_cols(cols, names=None) except ? NULL:
    col_count=len(cols)
    cdef gdf_column **c_cols = <gdf_column**>malloc(
        sizeof(gdf_column*) * col_count
    )

    cdef i
    for i in range(col_count):
        if names is None:
            name = None
        else:
            name = names[i]
        check_gdf_compatibility(cols[i])
        c_cols[i] = column_view_from_column(cols[i], name)

    return c_cols


cdef free_table(cudf_table* c_table, gdf_column** cols=NULL):
    cdef i
    for i in range(c_table[0].num_columns()):
        free_column(c_table[0].get_column(i))
    del c_table
    free(cols)


cdef free_column(gdf_column* c_col):
    if c_col is NULL:
        return
    free(c_col.col_name)
    free(c_col)


# gdf_context functions

_join_method_api = {
    'sort': GDF_SORT,
    'hash': GDF_HASH
}

_null_sort_behavior_api = {
    'null_as_largest': GDF_NULL_AS_LARGEST,
    'null_as_smallest': GDF_NULL_AS_SMALLEST
}


cdef gdf_context* create_context_view(
    flag_sorted,
    method,
    flag_distinct,
    flag_sort_result,
    flag_sort_inplace,
    flag_null_sort_behavior,
    flag_groupby_include_nulls
):

    cdef gdf_method method_api = _join_method_api[method]
    cdef gdf_context* context = <gdf_context*>malloc(sizeof(gdf_context))

    cdef int c_flag_sorted = flag_sorted
    cdef int c_flag_distinct = flag_distinct
    cdef int c_flag_sort_result = flag_sort_result
    cdef int c_flag_sort_inplace = flag_sort_inplace
    cdef gdf_null_sort_behavior nulls_sort_behavior_api = \
        _null_sort_behavior_api[flag_null_sort_behavior]
    cdef bool c_flag_groupby_include_nulls = flag_groupby_include_nulls

    with nogil:
        gdf_context_view(
            context,
            c_flag_sorted,
            method_api,
            c_flag_distinct,
            c_flag_sort_result,
            c_flag_sort_inplace,
            c_flag_groupby_include_nulls,
            nulls_sort_behavior_api,
        )

    return context


# # Error handling

cpdef check_gdf_error(errcode):
    """Get error message for the given error code.
    """
    cdef gdf_error c_errcode = errcode

    if c_errcode != GDF_SUCCESS:
        if c_errcode == GDF_CUDA_ERROR:
            with nogil:
                cudaerr = gdf_cuda_last_error()
                errname = gdf_cuda_error_name(cudaerr)
                details = gdf_cuda_error_string(cudaerr)
            msg = 'CUDA ERROR. {}: {}'.format(errname, details)

        else:
            with nogil:
                errname = gdf_error_get_name(c_errcode)
            msg = errname

        raise GDFError(errname, msg)


cpdef count_nonzero_mask(mask, size):
    """ Counts the number of null bits in a given validity mask
    """
    assert mask.size * mask_bitsize >= size
    cdef int nnz = 0
    cdef uintptr_t mask_ptr = get_ctype_ptr(mask)
    cdef int c_size = size

    if mask_ptr:
        with nogil:
            gdf_count_nonzero_mask(
                <valid_type*>mask_ptr,
                c_size,
                &nnz
            )

    return nnz


cdef char* py_to_c_str(object py_str) except? NULL:
    """
    Util to convert a Python bytes, bytearray, or unicode string to a char*,
    in a way that breaks free from the Cython garbage collector.
    """
    cdef char* c_str = NULL
    if py_str is not None:
        if isinstance(py_str, (str, unicode)):
            py_str = py_str.encode()
        elif not isinstance(py_str, (bytes, bytearray)):
            py_str = str(py_str).encode()
        py_str_len = len(py_str)
        c_str = <char*> malloc((py_str_len + 1) * sizeof(char))
        strcpy(c_str, py_str)
    return c_str
