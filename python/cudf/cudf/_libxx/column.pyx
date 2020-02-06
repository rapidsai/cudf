# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import pandas as pd
import cython
import rmm

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique

import cudf._libxx as libcudfxx
from cudf._libxx.lib cimport *

from cudf.core.buffer import Buffer
from cudf.utils.dtypes import is_categorical_dtype
from cudf.utils.utils import cached_property, calc_chunk_size, mask_bitsize


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
        self.size = size
        self.offset = offset
        self.dtype = dtype
        self.set_base_data(data)
        self.set_base_mask(mask)
        self.set_base_children(children)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if not pd.api.types.is_integer(value):
            raise TypeError("Expected an integer for size, got " +
                            type(value).__name__)

        if value > libcudfxx.MAX_COLUMN_SIZE:
            raise MemoryError(
                "Cannot create columns of size > {}. "
                "Consider using dask_cudf to partition your data".format(
                    libcudfxx.MAX_COLUMN_SIZE_STR)
            )

        try:
            del self.null_count
        except AttributeError:
            pass
        try:
            del self.children
        except AttributeError:
            pass
        if hasattr(self, "_indices"):
            self._indices = None
        if hasattr(self, "_nvcategory"):
            self._nvcategory = None
        if hasattr(self, "_nvstrings"):
            self._nvstrings = None

        self._size = int(value)

    @property
    def base_data(self):
        return self._base_data

    @cached_property
    def data(self):
        if self.offset == 0 or self.base_data is None:
            return self.base_data
        else:
            buf = Buffer(self.base_data)
            buf.ptr = buf.ptr + (self.offset * self.dtype.itemsize)
            buf.size = buf.size - (self.offset * self.dtype.itemsize)
            return buf

    @property
    def data_ptr(self):
        if self.data is None:
            return 0
        else:
            return self.data.ptr

    def set_base_data(self, value):
        if value is not None and not isinstance(value, Buffer):
            raise TypeError("Expected a Buffer or None for data, got " +
                            type(value).__name__)

        try:
            del self.data
        except AttributeError:
            pass

        self._base_data = value

    @property
    def nullable(self):
        return self.base_mask is not None

    @property
    def has_nulls(self):
        return self.null_count != 0

    @property
    def base_mask(self):
        return self._base_mask

    @cached_property
    def mask(self):
        mask = None
        if self.base_mask is not None:
            if self.offset == 0:
                mask = self.base_mask
            else:
                mask = libcudfxx.null_mask.copy_bitmask(self)
        return mask

    @property
    def mask_ptr(self):
        if self.mask is None:
            return 0
        else:
            return self.mask.ptr

    def set_base_mask(self, value):
        """
        Replaces the base mask buffer of the column inplace. This does not
        modify size or offset in any way, so the passed mask is expected to be
        compatible with the current offset.
        """
        if value is not None and not isinstance(value, Buffer):
            raise TypeError("Expected a Buffer or None for mask, got " +
                            type(value).__name__)
        if value is not None:
            offsetted_size = self.size + self.offset
            required_size = -(-self.size // mask_bitsize)  # ceiling divide
            if value.size < required_size:
                error_msg = (
                    "The Buffer for mask is smaller than expected, got " +
                    str(value.size) + " bytes, expected " + str(required_size)
                    + " bytes."
                )
                if self.offset != 0:
                    error_msg += (
                        " Note: the column has a non-zero offset and the mask "
                        "is expected to be sized according to the base "
                        "allocation as opposed to the offsetted allocation."
                    )
                raise RuntimeError(error_msg)

        try:
            del self.mask
        except AttributeError:
            pass
        try:
            del self.null_count
        except AttributeError:
            pass
        if hasattr(self, "_indices"):
            self._indices = None
        if hasattr(self, "_nvcategory"):
            self._nvcategory = None
        if hasattr(self, "_nvstrings"):
            self._nvstrings = None

        self._base_mask = value

    def set_mask(self, value):
        """
        Replaces the mask buffer of the column and returns a new column. This
        will zero the column offset, compute a new mask buffer if necessary,
        and compute new data Buffers zero-copy that use pointer arithmetic to
        properly adjust the pointer.
        """
        if value is not None and not isinstance(value, Buffer):
            raise TypeError("Expected a Buffer or None for mask, got " +
                            type(value).__name__)

        from cudf.core.column import build_column

        return build_column(
            self.data,
            self.dtype,
            value,
            self.size,
            offset=0,
            children=self.children
        )

    @cached_property
    def null_count(self):
        return self.compute_null_count()

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if not pd.api.types.is_integer(value):
            raise TypeError("Expected an integer for offset, got " +
                            type(value).__name__)

        try:
            del self.null_count
        except AttributeError:
            pass
        try:
            del self.mask
        except AttributeError:
            pass
        try:
            del self.children
        except AttributeError:
            pass
        if hasattr(self, "_indices"):
            self._indices = None
        if hasattr(self, "_nvcategory"):
            self._nvcategory = None
        if hasattr(self, "_nvstrings"):
            self._nvstrings = None

        if not hasattr(self, "_offset"):
            offset_diff = value
        else:
            offset_diff = value - self._offset

        self._offset = int(value)
        self.size = int(self.size - offset_diff)

    @property
    def base_children(self):
        return self._base_children

    @cached_property
    def children(self):
        return self._base_children

    def set_base_children(self, value):
        if not isinstance(value, tuple):
            raise TypeError("Expected a tuple of Columns for children, got " +
                            type(value).__name__)

        for child in value:
            if not isinstance(child, Column):
                raise TypeError(
                    "Expected each of children to be a  Column, got " +
                    type(child).__name__
                )

        try:
            del self.children
        except AttributeError:
            pass
        if hasattr(self, "_indices"):
            self._indices = None
        if hasattr(self, "_nvcategory"):
            self._nvcategory = None
        if hasattr(self, "_nvstrings"):
            self._nvstrings = None

        self._base_children = value

    def _mimic_inplace(self, other_col, inplace=False):
        """
        Given another column, update the attributes of this column to mimic an
        inplace operation. This does not modify the memory of Buffers, but
        instead replaces the Buffers and other attributes underneath the column
        object with the Buffers and attributes from the other column.
        """
        if inplace:
            self.size = other_col.size
            self.offset = other_col.offset
            self.dtype = other_col.dtype
            self.set_base_data(other_col.base_data)
            self.set_base_mask(other_col.base_mask)
            self.set_base_children(other_col.base_children)
            if hasattr(other_col, "_nvstrings"):
                self._nvstrings = other_col._nvstrings
            if hasattr(other_col, "_nvcategory"):
                self._nvcategory = other_col._nvcategory
            if hasattr(other_col, "_indices"):
                self._indices = other_col._indices
        else:
            return other_col

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

        data = <void*><uintptr_t>(col.base_data.ptr)

        cdef Column child_column
        if self.base_children:
            for child_column in self.base_children:
                children.push_back(child_column.mutable_view())

        cdef bitmask_type* mask
        if self.nullable:
            mask = <bitmask_type*><uintptr_t>(self.base_mask.ptr)
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

        data = <void*><uintptr_t>(col.base_data.ptr)

        cdef Column child_column
        if self.base_children:
            for child_column in self.base_children:
                children.push_back(child_column.view())

        cdef bitmask_type* mask
        if self.nullable:
            mask = <bitmask_type*><uintptr_t>(self.base_mask.ptr)
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
    cdef Column from_unique_ptr(unique_ptr[column] c_col):
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
        children = ()
        if c_children.size() != 0:
            children = tuple(Column.from_unique_ptr(move(c_children[i]))
                             for i in range(c_children.size()))

        return build_column(data, dtype=dtype, mask=mask, children=children)

    @staticmethod
    cdef Column from_column_view(column_view cv, object owner):
        from cudf.core.column import build_column

        column_owner = isinstance(owner, Column)

        size = cv.size()
        dtype = cudf_to_np_types[cv.type().id()]

        data_ptr = <uintptr_t>(cv.head[void]())
        data = None
        if data_ptr:
            data_owner = owner
            if column_owner:
                data_owner = owner.base_data
            data = Buffer(
                data=data_ptr,
                size=size * dtype.itemsize,
                owner=data_owner
            )

        mask_ptr = <uintptr_t>(cv.null_mask())
        mask = None
        if mask_ptr:
            mask_owner = owner
            if column_owner:
                mask_owner = owner.base_mask
            mask = Buffer(
                mask=mask_ptr,
                size=calc_chunk_size(size, mask_bitsize),
                owner=mask_owner
            )

        offset = cv.offset()

        children = []
        for child_index in range(cv.num_children()):
            child_owner = owner
            if column_owner:
                child_owner = owner.base_children[child_index]
            children.append(
                Column.from_column_view(
                    cv.child(child_index),
                    child_owner
                )
            )
        children = tuple(children)

        result = build_column(
            data,
            dtype,
            mask,
            offset,
            tuple(children)
        )
        result.size = size

        return result
