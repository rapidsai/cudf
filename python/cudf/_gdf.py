# Copyright (c) 2018, NVIDIA CORPORATION.

"""
This file provide binding to the libgdf library.
"""
import numpy as np
import pandas as pd

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm
import nvcategory

from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize


def unwrap_devary(devary):
    ptrval = devary.device_ctypes_pointer.value
    ptrval = ptrval or ffi.NULL   # replace None with NULL
    return ffi.cast('void*', ptrval)


def unwrap_mask(devary):
    ptrval = devary.device_ctypes_pointer.value
    ptrval = ptrval or ffi.NULL   # replace None with NULL
    return ffi.cast('gdf_valid_type*', ptrval), ptrval


def columnview_from_devary(devary, dtype=None):
    return _columnview(size=devary.size,  data=unwrap_devary(devary),
                       mask=ffi.NULL, dtype=dtype or devary.dtype,
                       null_count=0, nvcat=None)


def _columnview(size, data, mask, dtype, null_count, nvcat):
    colview = ffi.new('gdf_column*')
    extra_dtype_info = ffi.new('gdf_dtype_extra_info*')
    extra_dtype_info.time_unit = libgdf.TIME_UNIT_NONE
    if nvcat is not None:
        extra_dtype_info.category = ffi.cast('void*', nvcat.get_cpointer())
    else:
        extra_dtype_info.category = ffi.NULL

    if mask is None:
        null_count = 0
        mask = ffi.NULL

    libgdf.gdf_column_view_augmented(
        colview,
        data,
        mask,
        size,
        np_to_gdf_dtype(dtype),
        null_count,
        extra_dtype_info[0],
    )

    return colview


def columnview(size, data, mask=None, dtype=None, null_count=None,
               nvcat=None):
    """
    Make a column view.

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
    def unwrap(buffer):
        if buffer is None:
            return ffi.NULL
        assert buffer.mem.is_c_contiguous(), "libGDF expects contiguous memory"
        devary = buffer.to_gpu_array()
        return unwrap_devary(devary)

    if mask is not None:
        assert null_count is not None

    dtype = dtype or data.dtype
    if pd.api.types.is_categorical_dtype(dtype):
        dtype = data.dtype

    return _columnview(size=size, data=unwrap(data), mask=unwrap(mask),
                       dtype=dtype, null_count=null_count, nvcat=nvcat)


np_gdf_dict = {
    np.float64:      libgdf.GDF_FLOAT64,
    np.float32:      libgdf.GDF_FLOAT32,
    np.int64:        libgdf.GDF_INT64,
    np.int32:        libgdf.GDF_INT32,
    np.int16:        libgdf.GDF_INT16,
    np.int8:         libgdf.GDF_INT8,
    np.bool_:        libgdf.GDF_INT8,
    np.datetime64:   libgdf.GDF_DATE64,
    np.object_:      libgdf.GDF_STRING_CATEGORY,
    np.str_:         libgdf.GDF_STRING_CATEGORY,
    }


def np_to_gdf_dtype(dtype):
    """Util to convert numpy dtype to gdf dtype.
    """
    return np_gdf_dict[np.dtype(dtype).type]


def gdf_to_np_dtype(dtype):
    """Util to convert gdf dtype to numpy dtype.
    """
    return np.dtype({
         libgdf.GDF_FLOAT64: np.float64,
         libgdf.GDF_FLOAT32: np.float32,
         libgdf.GDF_INT64: np.int64,
         libgdf.GDF_INT32: np.int32,
         libgdf.GDF_INT16: np.int16,
         libgdf.GDF_INT8: np.int8,
         libgdf.GDF_DATE64: np.datetime64,
         libgdf.N_GDF_TYPES: np.int32,
         libgdf.GDF_CATEGORY: np.int32,
         libgdf.GDF_STRING_CATEGORY: np.object_,
     }[dtype])


_join_how_api = {
    'inner': libgdf.gdf_inner_join,
    'outer': libgdf.gdf_full_join,
    'left': libgdf.gdf_left_join,
}

_join_method_api = {
    'sort': libgdf.GDF_SORT,
    'hash': libgdf.GDF_HASH
}


def cffi_view_to_column_mem(cffi_view):
    gdf_dtype = cffi_view.dtype
    if gdf_dtype == libgdf.GDF_STRING_CATEGORY:
        data_ptr = int(ffi.cast("uintptr_t", cffi_view.data))
        # We need to create this just to make sure the memory is properly freed
        data = rmm.device_array_from_ptr(
            data_ptr,
            nelem=cffi_view.size,
            dtype='int32',
            finalizer=rmm._make_finalizer(data_ptr, 0)
        )
        nvcat_ptr = int(ffi.cast("uintptr_t", cffi_view.dtype_info.category))
        nvcat_obj = nvcategory.bind_cpointer(nvcat_ptr)
        nvstr_obj = nvcat_obj.to_strings()
        mask = None
        if cffi_view.valid:
            mask_ptr = int(ffi.cast("uintptr_t", cffi_view.valid))
            mask = rmm.device_array_from_ptr(
                    mask_ptr,
                    nelem=calc_chunk_size(cffi_view.size, mask_bitsize),
                    dtype=mask_dtype,
                    finalizer=rmm._make_finalizer(mask_ptr, 0)
                )
        return nvstr_obj, mask
    else:
        intaddr = int(ffi.cast("uintptr_t", cffi_view.data))
        data = rmm.device_array_from_ptr(
            intaddr,
            nelem=cffi_view.size,
            dtype=gdf_to_np_dtype(cffi_view.dtype),
            finalizer=rmm._make_finalizer(intaddr, 0)
        )
        mask = None
        if cffi_view.valid:
            intaddr = int(ffi.cast("uintptr_t", cffi_view.valid))
            mask = rmm.device_array_from_ptr(
                intaddr,
                nelem=calc_chunk_size(cffi_view.size, mask_bitsize),
                dtype=mask_dtype,
                finalizer=rmm._make_finalizer(intaddr, 0)
            )

        return data, mask


def rmm_initialize():
    rmm.initialize()
    return True


def rmm_finalize():
    rmm.finalize()
    return True
