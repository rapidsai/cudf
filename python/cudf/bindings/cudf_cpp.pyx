# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *

from .GDFError import GDFError

import numpy as np
import pandas as pd
import pyarrow as pa

cimport numpy as np

from cudf.utils import cudautils
from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free



dtypes = {np.float64:    GDF_FLOAT64,
          np.float32:    GDF_FLOAT32,
          np.int64:      GDF_INT64,
          np.int32:      GDF_INT32,
          np.int16:      GDF_INT16,
          np.int8:       GDF_INT8,
          np.bool_:      GDF_INT8,
          np.datetime64: GDF_DATE64}

def gdf_to_np_dtype(dtype):
    """Util to convert gdf dtype to numpy dtype.
    """
    return np.dtype({
         GDF_FLOAT64: np.float64,
         GDF_FLOAT32: np.float32,
         GDF_INT64: np.int64,
         GDF_INT32: np.int32,
         GDF_INT16: np.int16,
         GDF_INT8: np.int8,
         GDF_DATE64: np.datetime64,
         N_GDF_TYPES: np.int32,
         GDF_CATEGORY: np.int32,
     }[dtype])

def check_gdf_compatibility(col):
    """
    Raise TypeError when a column type does not have gdf support.
    """
    if not (col.dtype.type in dtypes or pd.api.types.is_categorical_dtype(col)):
        raise TypeError('column type `%s` not supported in gdf' % (col.dtype))


cpdef get_ctype_ptr(obj):
    return obj.device_ctypes_pointer.value

cpdef get_column_data_ptr(obj):
    return get_ctype_ptr(obj._data.mem)

cpdef get_column_valid_ptr(obj):
    return get_ctype_ptr(obj._mask.mem)


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

    if len(col) > 0:
        data_ptr = get_column_data_ptr(col)
    else:
        data_ptr = 0

    if col._mask is not None and col.null_count > 0:
        valid_ptr = get_column_valid_ptr(col)
    else:
        valid_ptr = 0

    if pd.api.types.is_categorical_dtype(col.dtype):
        g_dtype = GDF_INT8
    else:
        g_dtype = dtypes[col.dtype.type]

    cdef gdf_dtype c_dtype = g_dtype
    cdef gdf_size_type len_col = len(col)
    cdef gdf_size_type col_null_count = col.null_count

    with nogil:
        gdf_column_view_augmented(<gdf_column*>c_col,
                                <void*> data_ptr,
                                <gdf_valid_type*> valid_ptr,
                                len_col,
                                c_dtype,
                                col_null_count)


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
            g_dtype = GDF_INT8
        elif dtype != np.bool_:
            g_dtype = dtypes[dtype.type]
        else:
            print("HITHER ::::::")
            g_dtype = dtypes[dtype]
    else:
        g_dtype = dtypes[data.dtype]

    if null_count is None:
        null_count = 0

    cdef gdf_dtype c_dtype = g_dtype
    cdef gdf_size_type c_size = size
    cdef gdf_size_type c_null_count = null_count
    
    with nogil:
        gdf_column_view_augmented(<gdf_column*>c_col,
                                <void*> data_ptr,
                                <gdf_valid_type*> valid_ptr,
                                c_size,
                                c_dtype,
                                c_null_count)

    return c_col


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





