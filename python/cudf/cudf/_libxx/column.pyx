import numpy as np

import rmm

from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool

cimport cudf._lib.cudf as gdf
import cudf._lib.cudf as gdf

from cudf._libxx.lib cimport *
from cudf.core.buffer import Buffer
from libc.stdlib cimport malloc, free

from cudf.utils.dtypes import is_categorical_dtype

np_to_cudf_types = {np.dtype('int16'): INT32,
                    np.dtype('int32'): INT32,
                    np.dtype('int64'): INT64,
                    np.dtype('float32'): FLOAT32,
                    np.dtype('float64'): FLOAT64,
                    np.dtype("datetime64[D]"): TIMESTAMP_DAYS,
                    np.dtype("datetime64[s]"): TIMESTAMP_SECONDS,
                    np.dtype("datetime64[ms]"): TIMESTAMP_MILLISECONDS,
                    np.dtype("datetime64[us]"): TIMESTAMP_MICROSECONDS,
                    np.dtype("datetime64[ns]"): TIMESTAMP_NANOSECONDS,
}

cudf_to_np_types = {INT16: np.dtype('int16'),
                    INT32: np.dtype('int32'),
                    INT64: np.dtype('int64'),
                    FLOAT32: np.dtype('float32'),
                    FLOAT64: np.dtype('float64'),
                    TIMESTAMP_DAYS: np.dtype("datetime64[D]"),
                    TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
                    TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
                    TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
                    TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
}

cdef class Column:
    def __init__(self, data, size, dtype, mask=None):
        self.data = data
        self.size = size
        self.dtype = dtype
        self.mask = mask

    @property
    def null_count(self):
        return self.null_count()

    cdef size_type null_count(self):
        return self.view().null_count()

    cdef mutable_column_view mutable_view(self) except *:
        cdef type_id tid = np_to_cudf_types[np.dtype(self.dtype)]
        cdef data_type dtype = data_type(tid)
        cdef void* data = <void*><uintptr_t>(self.data.ptr)
        cdef bitmask_type* mask
        if self.mask is not None:
            mask = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            mask = NULL
        return mutable_column_view(
            dtype,
            self.size,
            data,
            mask)

    cdef column_view view(self) except *:
        cdef type_id tid = np_to_cudf_types[np.dtype(self.dtype)]
        cdef data_type dtype = data_type(tid)
        cdef void* data = <void*><uintptr_t>(self.data.ptr)
        cdef bitmask_type* mask
        if self.mask is not None:
            mask = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            mask = NULL
        return column_view(
            dtype,
            self.size,
            data,
            mask)

    @staticmethod
    cdef Column from_ptr(unique_ptr[column] c_col):
        from cudf.core.column import build_column

        size = c_col.get()[0].size()
        dtype = cudf_to_np_types[c_col.get()[0].type().id()]
        has_nulls = c_col.get()[0].has_nulls()
        cdef column_contents contents = c_col.get()[0].release()
        data = DeviceBuffer.from_ptr(contents.data.release())
        if has_nulls:
            mask = DeviceBuffer.from_ptr(contents.null_mask.release())
        else:
            mask = None
        return build_column(data, dtype=dtype, mask=mask)


    cdef gdf.gdf_column* gdf_column_view(self) except *:
        cdef gdf.gdf_column* c_col = <gdf.gdf_column*>malloc(sizeof(gdf.gdf_column))
        cdef uintptr_t data_ptr
        cdef uintptr_t valid_ptr
        cdef uintptr_t category
        cdef gdf.gdf_dtype c_dtype = gdf.dtypes[self.dtype.type]

        if c_dtype == gdf.GDF_STRING_CATEGORY:
            raise NotImplementedError
        else:
            category = 0
            if len(self) > 0:
                data_ptr = self.data.ptr
            else:
                data_ptr = 0

        if self.mask:
            valid_ptr = self.mask.ptr
        else:
            valid_ptr = 0

        cdef char* c_col_name = gdf.py_to_c_str(self.name)
        cdef size_type len_col = len(self)
        cdef size_type c_null_count = self.null_count()
        cdef gdf.gdf_time_unit c_time_unit = gdf.np_dtype_to_gdf_time_unit(self.dtype)
        cdef gdf.gdf_dtype_extra_info c_extra_dtype_info = gdf.gdf_dtype_extra_info(
            time_unit=c_time_unit,
            category=<void*>category
        )

        with nogil:
            gdf.gdf_column_view_augmented(
                <gdf.gdf_column*>c_col,
                <void*>data_ptr,
                <gdf.valid_type*>valid_ptr,
                len_col,
                c_dtype,
                c_null_count,
                c_extra_dtype_info,
                c_col_name
            )

        return c_col

    @staticmethod
    cdef Column from_gdf_column(gdf.gdf_column* c_col):

        from cudf.core.column import build_column
        from cudf.utils.utils import mask_bitsize, calc_chunk_size

        gdf_dtype = c_col.dtype
        data_ptr = int(<uintptr_t>c_col.data)

        if gdf_dtype == gdf.GDF_STRING:
            raise NotImplementedError
        elif gdf_dtype == gdf.GDF_STRING_CATEGORY:
            raise NotImplementedError
        else:
            dtype = np.dtype(gdf.gdf_dtypes[gdf_dtype])
            dbuf = rmm.DeviceBuffer(
                ptr=data_ptr,
                size=dtype.itemsize * c_col.size,
            )
            data = Buffer.from_device_buffer(dbuf)
        mask = None
        if c_col.valid:
            mask_ptr = int(<uintptr_t>c_col.valid)
            mbuf = rmm.DeviceBuffer(
                ptr=mask_ptr,
                size=calc_chunk_size(c_col.size,mask_bitsize)
            )
            mask = Buffer.from_device_buffer(mbuf)

        return build_column(data=data,
                            dtype=dtype,
                            mask=mask)


    cdef gdf.gdf_dtype gdf_type(self) except? gdf.GDF_invalid:
        dtype = self.dtype

        # if dtype is pd.CategoricalDtype, use the codes' gdf_dtype
        if is_categorical_dtype(dtype):
            if self.data is None:
                raise NotImplementedError
            return self.data.gdf_dtype_from_value()
        # if dtype is np.datetime64, interrogate the dtype's time_unit resolution
        if dtype.kind == 'M':
            time_unit, _ = np.datetime_data(dtype)
            if time_unit in gdf.np_to_gdf_time_unit:
                # time_unit is valid so must be a GDF_TIMESTAMP
                return gdf.GDF_TIMESTAMP
            # else default to GDF_DATE64
            return gdf.GDF_DATE64
        # everything else is a 1-1 mapping
        if dtype.type in gdf.dtypes:
            return gdf.dtypes[dtype.type]
        raise TypeError('cannot convert numpy dtype `%s` to gdf_dtype' % (dtype))


