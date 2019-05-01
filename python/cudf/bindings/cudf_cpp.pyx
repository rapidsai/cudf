# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.GDFError import GDFError
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import numpy as np
import pandas as pd
import pyarrow as pa

from cudf.utils import cudautils
from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize
from librmm_cffi import librmm as rmm
import nvstrings
import nvcategory


dtypes = {
    np.float64:    GDF_FLOAT64,
    np.float32:    GDF_FLOAT32,
    np.int64:      GDF_INT64,
    np.int32:      GDF_INT32,
    np.int16:      GDF_INT16,
    np.int8:       GDF_INT8,
    np.bool_:      GDF_BOOL8,
    np.datetime64: GDF_DATE64,
    np.object_:    GDF_STRING_CATEGORY,
    np.str_:       GDF_STRING_CATEGORY,
}

gdf_dtypes = {
    GDF_FLOAT64:           np.float64,
    GDF_FLOAT32:           np.float32,
    GDF_INT64:             np.int64,
    GDF_INT32:             np.int32,
    GDF_INT16:             np.int16,
    GDF_INT8:              np.int8,
    GDF_BOOL8:             np.bool_,
    GDF_DATE64:            np.datetime64,
    GDF_CATEGORY:          np.int32,
    GDF_STRING_CATEGORY:   np.object_,
    GDF_STRING:            np.object_,
    N_GDF_TYPES:           np.int32
}

np_pa_dtypes = {
    np.float64:     pa.float64(),
    np.float32:     pa.float32(),
    np.int64:       pa.int64(),
    np.int32:       pa.int32(),
    np.int16:       pa.int16(),
    np.int8:        pa.int8(),
    np.bool_:       pa.int8(),
    np.datetime64:  pa.date64(),
    np.object_:     pa.string(),
    np.str_:        pa.string(),
}

def gdf_to_np_dtype(dtype):
    """Util to convert gdf dtype to numpy dtype.
    """
    return np.dtype(gdf_dtypes[dtype])

def np_to_pa_dtype(dtype):
    """Util to convert numpy dtype to PyArrow dtype
    """
    return np_pa_dtypes[np.dtype(dtype).type]

def check_gdf_compatibility(col):
    """
    Raise TypeError when a column type does not have gdf support.
    """
    if not (col.dtype.type in dtypes or pd.api.types.is_categorical_dtype(col)):
        raise TypeError('column type `%s` not supported in gdf' % (col.dtype))


cpdef get_ctype_ptr(obj):
    if obj.device_ctypes_pointer.value is None:
        return 0
    else:
        return obj.device_ctypes_pointer.value

cpdef get_column_data_ptr(obj):
    return get_ctype_ptr(obj._data.mem)

cpdef get_column_valid_ptr(obj):
    return get_ctype_ptr(obj._mask.mem)

cdef gdf_dtype get_dtype(dtype):
    return dtypes[dtype]

cdef get_scalar_value(gdf_scalar scalar):
    """
    Returns typed value from a gdf_scalar
    0-dim array is retuned if dtype is date32/64, timestamp
    """
    return {
        GDF_FLOAT64: scalar.data.fp64,
        GDF_FLOAT32: scalar.data.fp32,
        GDF_INT64:   scalar.data.si64,
        GDF_INT32:   scalar.data.si32,
        GDF_INT16:   scalar.data.si16,
        GDF_INT8:    scalar.data.si08,
        GDF_BOOL8:   np.array(scalar.data.b08).astype(np.bool_),
        GDF_DATE32:  np.array(scalar.data.dt32).astype('datetime64[D]'),
        GDF_DATE64:  np.array(scalar.data.dt64).astype('datetime64[ms]'),
        GDF_TIMESTAMP: np.array(scalar.data.tmst).astype('datetime64[ns]'),
    }[scalar.dtype]

# gdf_column functions

cdef gdf_column* column_view_from_column(col):
    """
    Make a column view from a column

    Parameters
    ----------
    size : int
        Data count.
    data : Buffer
        The data buffer.
    mask : Buffer; optional
        The mask buffer.
    dtype : numpy.dtype; optional
        The dtype of the data.  Defaults to *data.dtype*.
    """

    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr
    cdef uintptr_t valid_ptr
    cdef uintptr_t category

    if pd.api.types.is_categorical_dtype(col.dtype):
        g_dtype = dtypes[col.data.dtype.type]
    else:
        g_dtype = dtypes[col.dtype.type]

    if g_dtype == GDF_STRING_CATEGORY:
        category = col.nvcategory.get_cpointer()
        if len(col) > 0:
            data_ptr = get_ctype_ptr(col.indices.mem)
        else:
            data_ptr = 0
    else:
        category = 0

        if len(col) > 0:
            data_ptr = get_column_data_ptr(col)
        else:
            data_ptr = 0

    if col._mask is not None and col.null_count > 0:
        valid_ptr = get_column_valid_ptr(col)
    else:
        valid_ptr = 0

    cdef gdf_dtype c_dtype = g_dtype
    cdef gdf_size_type len_col = len(col)
    cdef gdf_size_type c_null_count = col.null_count
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(
        time_unit = TIME_UNIT_NONE,
        category = <void*> category
    )

    with nogil:
        gdf_column_view_augmented(<gdf_column*>c_col,
                                <void*> data_ptr,
                                <gdf_valid_type*> valid_ptr,
                                len_col,
                                c_dtype,
                                c_null_count,
                                c_extra_dtype_info)


    return c_col


cdef gdf_column* column_view_from_NDArrays(size, data, mask, dtype,
                                           null_count):
    """
    Make a column view from NDArrays

    Parameters
    ----------
    size : int
        Data count.
    data : Buffer
        The data buffer.
    mask : Buffer; optional
        The mask buffer.
    dtype : numpy.dtype; optional
        The dtype of the data.  Defaults to *data.dtype*.
    """
    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr
    cdef uintptr_t valid_ptr

    if data is not None:
        data_ptr = get_ctype_ptr(data)
    else:
        data_ptr = 0

    if mask is not None:
        valid_ptr = get_ctype_ptr(mask)
    else:
        valid_ptr = 0

    if dtype is not None:
        if pd.api.types.is_categorical_dtype(dtype):
            if data is None:
                g_dtype = dtypes[np.int8]
            else:
                g_dtype = dtypes[data.dtype.type]
        elif dtype != np.bool_:
            g_dtype = dtypes[dtype.type]
        else:
            g_dtype = dtypes[dtype]
    else:
        g_dtype = dtypes[data.dtype]

    if null_count is None:
        null_count = 0

    cdef gdf_dtype c_dtype = g_dtype
    cdef gdf_size_type c_size = size
    cdef gdf_size_type c_null_count = null_count
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(
        time_unit = TIME_UNIT_NONE,
        category = <void*> 0
    )
    
    with nogil:
        gdf_column_view_augmented(<gdf_column*>c_col,
                                <void*> data_ptr,
                                <gdf_valid_type*> valid_ptr,
                                c_size,
                                c_dtype,
                                c_null_count,
                                c_extra_dtype_info)

    return c_col


cdef gdf_column_to_column_mem(gdf_column* input_col):
    gdf_dtype = input_col.dtype
    data_ptr = int(<uintptr_t>input_col.data)
    if gdf_dtype == GDF_STRING:
        data = nvstrings.bind_cpointer(data_ptr)
    elif gdf_dtype == GDF_STRING_CATEGORY:
        # Need to do this just to make sure it's freed properly
        garbage = rmm.device_array_from_ptr(
            data_ptr,
            nelem=input_col.size,
            dtype='int32',
            finalizer=rmm._make_finalizer(data_ptr, 0)
        )
        nvcat_ptr = int(<uintptr_t>input_col.dtype_info.category)
        nvcat_obj = nvcategory.bind_cpointer(nvcat_ptr)
        data = nvcat_obj.to_strings()
    else:
        data = rmm.device_array_from_ptr(
            data_ptr,
            nelem=input_col.size,
            dtype=gdf_to_np_dtype(input_col.dtype),
            finalizer=rmm._make_finalizer(data_ptr, 0)
        )

    mask = None
    if input_col.valid:
        mask_ptr = int(<uintptr_t>input_col.valid)
        mask = rmm.device_array_from_ptr(
            mask_ptr,
            nelem=calc_chunk_size(input_col.size, mask_bitsize),
            dtype=mask_dtype,
            finalizer=rmm._make_finalizer(mask_ptr, 0)
        )

    return data, mask


cdef update_nvstrings_col(col, uintptr_t category_ptr):
    nvcat_ptr = int(category_ptr)
    nvcat_obj = None
    if nvcat_ptr:
        nvcat_obj = nvcategory.bind_cpointer(nvcat_ptr)
        nvstr_obj = nvcat_obj.to_strings()
    else:
        nvstr_obj = nvstrings.to_device([])
    col._data = nvstr_obj
    col._nvcategory = nvcat_obj

# gdf_context functions

_join_method_api = {
    'sort': GDF_SORT,
    'hash': GDF_HASH
}

cdef gdf_context* create_context_view(flag_sorted, method, flag_distinct,
                                 flag_sort_result, flag_sort_inplace):

    cdef gdf_method method_api = _join_method_api[method]
    cdef gdf_context* context = <gdf_context*>malloc(sizeof(gdf_context))

    cdef int c_flag_sorted = flag_sorted
    cdef int c_flag_distinct = flag_distinct
    cdef int c_flag_sort_result = flag_sort_result
    cdef int c_flag_sort_inplace = flag_sort_inplace
    
    with nogil:
        gdf_context_view(context,
                         c_flag_sorted,
                         method_api,
                         c_flag_distinct,
                         c_flag_sort_result,
                         c_flag_sort_inplace)

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
                <gdf_valid_type*>mask_ptr,
                c_size,
                &nnz
            )

    return nnz
