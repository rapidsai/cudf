# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import pandas as pd
import cython
import rmm

from cpython.buffer cimport PyObject_CheckBuffer
from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique

import cudf._libxx as libcudfxx
from cudf._libxx.lib cimport *
from cudf._libxx.null_mask import bitmask_allocation_size_bytes

from cudf.core.buffer import Buffer
from cudf.utils.dtypes import is_categorical_dtype


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
        if not pd.api.types.is_integer(offset):
            raise TypeError("Expected an integer for offset, got " +
                            type(offset).__name__)

        if not pd.api.types.is_integer(size):
            raise TypeError("Expected an integer for size, got " +
                            type(size).__name__)

        if size < 0:
            raise RuntimeError(
                "Cannot create columns of size < 0. Got size: {}".format(
                    str(size)
                )
            )

        if size > libcudfxx.MAX_COLUMN_SIZE:
            raise MemoryError(
                "Cannot create columns of size > {}. "
                "Consider using dask_cudf to partition your data".format(
                    libcudfxx.MAX_COLUMN_SIZE_STR
                )
            )

        self._offset = int(offset)
        self._size = int(size)
        self.dtype = dtype
        self.set_base_children(children)
        self.set_base_data(data)
        self.set_base_mask(mask)

    @property
    def base_size(self):
        return int(self.base_data.size / self.dtype.itemsize)

    @property
    def size(self):
        return self._size

    @property
    def base_data(self):
        return self._base_data

    @property
    def base_data_ptr(self):
        if self.base_data is None:
            return 0
        else:
            return self.base_data.ptr

    @property
    def data(self):
        if self._data is None:
            if self.base_data is None:
                self._data = self.base_data
            else:
                buf = Buffer(self.base_data)
                buf.ptr = buf.ptr + (self.offset * self.dtype.itemsize)
                buf.size = self.size * self.dtype.itemsize
                self._data = buf
        return self._data

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

        self._data = None

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

    @property
    def base_mask_ptr(self):
        if self.base_mask is None:
            return 0
        else:
            return self.base_mask.ptr

    @property
    def mask(self):
        if self._mask is None:
            if self.base_mask is None or self.offset == 0:
                self._mask = self.base_mask
            else:
                self._mask = libcudfxx.null_mask.copy_bitmask(self)
        return self._mask

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
            required_size = bitmask_allocation_size_bytes(self.base_size)
            if value.size < required_size:
                error_msg = (
                    "The Buffer for mask is smaller than expected, got " +
                    str(value.size) + " bytes, expected " +
                    str(required_size) + " bytes."
                )
                if self.offset > 0 or self.size < self.base_size:
                    error_msg += (
                        "\n\nNote: The mask is expected to be sized according "
                        "to the base allocation as opposed to the offsetted or"
                        " sized allocation."
                    )
                raise ValueError(error_msg)

        self._mask = None
        self._null_count = None
        self._children = None
        self._base_mask = value

    def set_mask(self, value):
        """
        Replaces the mask buffer of the column and returns a new column. This
        will zero the column offset, compute a new mask buffer if necessary,
        and compute new data Buffers zero-copy that use pointer arithmetic to
        properly adjust the pointer.
        """
        mask_size = bitmask_allocation_size_bytes(self.size)
        required_num_bytes = -(-self.size // 8)  # ceiling divide
        error_msg = (
            "The value for mask is smaller than expected, got {}  bytes, "
            "expected " + str(required_num_bytes) + " bytes."
        )
        if value is None:
            mask = None
        elif hasattr(value, "__cuda_array_interface__"):
            mask = Buffer(value)
            if mask.size < required_num_bytes:
                raise ValueError(error_msg.format(str(value.size)))
            if mask.size < mask_size:
                dbuf = rmm.DeviceBuffer(size=mask_size)
                dbuf.copy_from_device(value)
                mask = Buffer(dbuf)
        elif hasattr(value, "__array_interface__"):
            value = np.asarray(value).view("u1")[:mask_size]
            if value.size < required_num_bytes:
                raise ValueError(error_msg.format(str(value.size)))
            dbuf = rmm.DeviceBuffer(size=mask_size)
            dbuf.copy_from_host(value)
            mask = Buffer(dbuf)
        elif PyObject_CheckBuffer(value):
            value = np.asarray(value).view("u1")[:mask_size]
            if value.size < required_num_bytes:
                raise ValueError(error_msg.format(str(value.size)))
            dbuf = rmm.DeviceBuffer(size=mask_size)
            dbuf.copy_from_host(value)
            mask = Buffer(dbuf)
        else:
            raise TypeError(
                "Expected a Buffer-like object or None for mask, got "
                + type(value).__name__
            )

        from cudf.core.column import build_column

        return build_column(
            self.data,
            self.dtype,
            mask,
            self.size,
            offset=0,
            children=self.children
        )

    @property
    def null_count(self):
        if self._null_count is None:
            self._null_count = self.compute_null_count()
        return self._null_count

    @property
    def offset(self):
        return self._offset

    @property
    def base_children(self):
        return self._base_children

    @property
    def children(self):
        if self._children is None:
            self._children = self.base_children
        return self._children

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

        self._children = None
        self._base_children = value

    def _mimic_inplace(self, other_col, inplace=False):
        """
        Given another column, update the attributes of this column to mimic an
        inplace operation. This does not modify the memory of Buffers, but
        instead replaces the Buffers and other attributes underneath the column
        object with the Buffers and attributes from the other column.
        """
        if inplace:
            self._offset = other_col.offset
            self._size = other_col.size
            self.dtype = other_col.dtype
            self.set_base_data(other_col.base_data)
            self.set_base_mask(other_col.base_mask)
            self.set_base_children(other_col.base_children)
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

        data = <void*><uintptr_t>(col.base_data_ptr)

        cdef Column child_column
        if self.base_children:
            for child_column in self.base_children:
                children.push_back(child_column.mutable_view())

        cdef bitmask_type* mask
        if self.nullable:
            mask = <bitmask_type*><uintptr_t>(self.base_mask_ptr)
        else:
            mask = NULL

        null_count = self.null_count
        if null_count is None:
            null_count = UNKNOWN_NULL_COUNT
        cdef size_type c_null_count = null_count

        return mutable_column_view(
            dtype,
            self.size,
            data,
            mask,
            c_null_count,
            offset,
            children)

    cdef column_view view(self) except *:
        null_count = self.null_count
        if null_count is None:
            null_count = UNKNOWN_NULL_COUNT
        cdef size_type c_null_count = null_count
        return self._view(c_null_count)

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

        data = <void*><uintptr_t>(col.base_data_ptr)

        cdef Column child_column
        if self.base_children:
            for child_column in self.base_children:
                children.push_back(child_column.view())

        cdef bitmask_type* mask
        if self.nullable:
            mask = <bitmask_type*><uintptr_t>(self.base_mask_ptr)
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
        """
        Given a ``cudf::column_view``, constructs a ``cudf.Column`` from it,
        along with referencing an ``owner`` Python object that owns the memory
        lifetime. If ``owner`` is a ``cudf.Column``, we reach inside of it and
        make the owner of each newly created ``Buffer`` the respective
        ``Buffer`` from the ``owner`` ``cudf.Column``. If ``owner`` is
        ``None``, we allocate new memory for the resulting ``cudf.Column``.
        """
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
            if data_owner is None:
                data = Buffer(
                    rmm.DeviceBuffer(ptr=data_ptr, size=size * dtype.itemsize)
                )
            else:
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
            if mask_owner is None:
                mask = Buffer(
                    rmm.DeviceBuffer(
                        ptr=mask_ptr,
                        size=bitmask_allocation_size_bytes(size)
                    )
                )
            else:
                mask = Buffer(
                    mask=mask_ptr,
                    size=bitmask_allocation_size_bytes(size),
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
            size,
            offset,
            tuple(children)
        )

        return result
