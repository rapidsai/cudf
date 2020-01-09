import numpy as np
import pandas as pd
import cython
import rmm

from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._libxx.lib cimport *
from cudf.core.buffer import Buffer
from libc.stdlib cimport malloc, free

from cudf.utils.dtypes import is_categorical_dtype
from cudf.utils.utils import cached_property


np_to_cudf_types = {
    np.dtype('int8'): INT8,
    np.dtype('int16'): INT16,
    np.dtype('int32'): INT32,
    np.dtype('int64'): INT64,
    np.dtype('float32'): FLOAT32,
    np.dtype('float64'): FLOAT64,
    np.dtype("datetime64[s]"): TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): TIMESTAMP_NANOSECONDS,
    np.dtype("object"): STRING,
    np.dtype("bool"): BOOL8
}

cudf_to_np_types = {
    INT8: np.dtype('int8'),
    INT16: np.dtype('int16'),
    INT32: np.dtype('int32'),
    INT64: np.dtype('int64'),
    FLOAT32: np.dtype('float32'),
    FLOAT64: np.dtype('float64'),
    TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    STRING: np.dtype("object"),
    BOOL8: np.dtype("bool")
}


@cython.auto_pickle(True)
cdef class Column:
    """
    A Column stores columnar data in device memory.
    A Column may be composed of:

    * A *data* Buffer
    * One or more (optional) *children* Columns
    * An (optional) *mask* Buffer representing the nullmask

    The *dtype* indicates the Column's element type.
    """

    def __init__(self, data, size, dtype, mask=None, offset=0, children=()):
        if not isinstance(data, Buffer):
            raise TypeError("Expected a Buffer for data, got " +
                            type(data).__name__)
        if mask is not None and not isinstance(mask, Buffer):
            raise TypeError("Expected a Buffer for mask, got " +
                            type(mask).__name__)
        if not pd.api.types.is_integer(size):
            raise TypeError("Expected an integer for size, got " +
                            type(size).__name__)
        if not pd.api.types.is_integer(offset):
            raise TypeError("Expected an integer for offset, got " +
                            type(offset).__name__)
        if not isinstance(children, tuple):
            raise TypeError("Expected a tuple of Columns for children, got " +
                            type(children).__name__)

        for child in children:
            if not isinstance(child, Column):
                raise TypeError(
                    "Expected each of children to be a  Column, got " +
                    type(child).__name__
                )

        self._data = data
        self.size = size
        self.dtype = dtype
        self._mask = mask
        self.offset = offset
        self.children = children

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is not None:
            if not isinstance(value, Buffer):
                raise TypeError("data must be a Buffer or None")
        self._data = value

    @property
    def nullable(self):
        return self.mask is not None

    @property
    def has_nulls(self):
        return self.null_count != 0

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if value is not None:
            if not isinstance(value, Buffer):
                raise TypeError("mask must be a Buffer or None")
        self._set_mask(value)

    def _set_mask(self, value):
        self._mask = value
        if hasattr(self, "null_count"):
            del self.null_count

    @cached_property
    def null_count(self):
        return self.compute_null_count()

    cdef size_type compute_null_count(self) except? 0:
        return self._view(UNKNOWN_NULL_COUNT).null_count()

    cdef mutable_column_view mutable_view(self) except *:
        if is_categorical_dtype(self.dtype):
            col = self.codes
        else:
            col = self
        data_dtype = col.dtype

        cdef type_id tid = np_to_cudf_types[np.dtype(data_dtype)]
        cdef data_type dtype = data_type(tid)
        cdef size_type offset = self.offset
        cdef vector[mutable_column_view] children
        cdef void* data

        data = <void*><uintptr_t>(col.data.ptr)

        cdef Column child_column
        if self.children:
            for child_column in self.children:
                children.push_back(child_column.mutable_view())

        cdef bitmask_type* mask
        if self.nullable:
            mask = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            mask = NULL

        cdef size_type c_null_count = self.null_count

        return mutable_column_view(
            dtype,
            self.size,
            data,
            mask,
            c_null_count,
            offset,
            children)

    cdef column_view view(self) except *:
        return self._view(self.null_count)

    cdef column_view _view(self, size_type null_count) except *:
        if is_categorical_dtype(self.dtype):
            col = self.codes
        else:
            col = self
        data_dtype = col.dtype
        cdef type_id tid = np_to_cudf_types[np.dtype(data_dtype)]
        cdef data_type dtype = data_type(tid)
        cdef size_type offset = self.offset
        cdef vector[column_view] children
        cdef void* data

        data = <void*><uintptr_t>(col.data.ptr)

        cdef Column child_column
        if self.children:
            for child_column in self.children:
                children.push_back(child_column.view())

        cdef bitmask_type* mask
        if self.nullable:
            mask = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            mask = NULL

        cdef size_type c_null_count = null_count

        return column_view(
            dtype,
            self.size,
            data,
            mask,
            c_null_count,
            offset,
            children)

    @staticmethod
    cdef Column from_ptr(unique_ptr[column] c_col):
        from cudf.core.column import build_column

        size = c_col.get()[0].size()
        dtype = cudf_to_np_types[c_col.get()[0].type().id()]
        has_nulls = c_col.get()[0].has_nulls()

        # After call to release(), c_col is unusable
        cdef column_contents contents = c_col.get()[0].release()

        data = DeviceBuffer.c_from_unique_ptr(move(contents.data))
        data = Buffer(data)

        if has_nulls:
            mask = DeviceBuffer.c_from_unique_ptr(move(contents.null_mask))
            mask = Buffer(mask)
        else:
            mask = None

        cdef vector[unique_ptr[column]] c_children = move(contents.children)
        children = None
        if c_children.size() != 0:
            children = tuple(Column.from_ptr(move(c_children[i]))
                             for i in range(c_children.size()))

        return build_column(data, dtype=dtype, mask=mask, children=children)
