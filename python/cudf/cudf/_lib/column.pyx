# Copyright (c) 2020, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import rmm

import cudf

from cudf.core.buffer import Buffer
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_list_dtype,
    is_struct_dtype
)
import cudf._lib as libcudfxx

from cpython.buffer cimport PyObject_CheckBuffer
from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.vector cimport vector
from libcpp.utility cimport move
from cudf._lib.cpp.strings.convert.convert_integers cimport (
    from_integers as cpp_from_integers
)

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.types import np_to_cudf_types, cudf_to_np_types
from cudf._lib.types cimport (
    underlying_type_t_type_id,
    dtype_from_column_view
)
from cudf._lib.null_mask import bitmask_allocation_size_bytes

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column_factories cimport (
    make_column_from_scalar as cpp_make_column_from_scalar,
    make_numeric_column
)
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.scalar cimport Scalar
cimport cudf._lib.cpp.types as libcudf_types
cimport cudf._lib.cpp.unary as libcudf_unary

cdef class Column:
    """
    A Column stores columnar data in device memory.
    A Column may be composed of:

    * A *data* Buffer
    * One or more (optional) *children* Columns
    * An (optional) *mask* Buffer representing the nullmask

    The *dtype* indicates the Column's element type.
    """
    def __init__(
            self,
            object data,
            int size,
            object dtype,
            object mask=None,
            int offset=0,
            object null_count=None,
            object children=()
    ):

        self._size = size
        self._dtype = dtype
        self._offset = offset
        self._null_count = null_count
        self.set_base_children(children)
        self.set_base_data(data)
        self.set_base_mask(mask)

    @property
    def base_size(self):
        return int(self.base_data.size / self.dtype.itemsize)

    @property
    def dtype(self):
        return self._dtype

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
                if self.size == 0:
                    buf.ptr = 0
                else:
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
            if value.__cuda_array_interface__["typestr"] not in ("|i1", "|u1"):
                if isinstance(value, Column):
                    value = value.data_array_view
                value = cp.asarray(value).view('|u1')
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

        return cudf.core.column.build_column(
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
        if (self.offset == 0) and (self.size == self.base_size):
            self._children = self.base_children
        if self._children is None:
            if self.base_children == ():
                self._children = ()
            else:
                self._children = Column.from_unique_ptr(
                    make_unique[column](self.view())
                ).base_children
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
            self._dtype = other_col._dtype
            self.set_base_data(other_col.base_data)
            self.set_base_mask(other_col.base_mask)
            self.set_base_children(other_col.base_children)
        else:
            return other_col

    cdef libcudf_types.size_type compute_null_count(self) except? 0:
        return self._view(libcudf_types.UNKNOWN_NULL_COUNT).null_count()

    cdef mutable_column_view mutable_view(self) except *:
        if is_categorical_dtype(self.dtype):
            col = self.base_children[0]
        else:
            col = self
        data_dtype = col.dtype

        cdef libcudf_types.type_id tid = <libcudf_types.type_id> (
            <underlying_type_t_type_id> (
                np_to_cudf_types[np.dtype(data_dtype)]
            )
        )
        cdef libcudf_types.data_type dtype = libcudf_types.data_type(tid)
        cdef libcudf_types.size_type offset = self.offset
        cdef vector[mutable_column_view] children
        cdef void* data

        data = <void*><uintptr_t>(col.base_data_ptr)

        cdef Column child_column
        if col.base_children:
            for child_column in col.base_children:
                children.push_back(child_column.mutable_view())

        cdef libcudf_types.bitmask_type* mask
        if self.nullable:
            mask = <libcudf_types.bitmask_type*><uintptr_t>(self.base_mask_ptr)
        else:
            mask = NULL

        null_count = self._null_count

        if null_count is None:
            null_count = libcudf_types.UNKNOWN_NULL_COUNT
        cdef libcudf_types.size_type c_null_count = null_count

        self._mask = None
        self._null_count = None
        self._children = None
        self._data = None

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
            null_count = libcudf_types.UNKNOWN_NULL_COUNT
        cdef libcudf_types.size_type c_null_count = null_count
        return self._view(c_null_count)

    cdef column_view _view(self, libcudf_types.size_type null_count) except *:
        if is_categorical_dtype(self.dtype):
            col = self.base_children[0]
        else:
            col = self

        data_dtype = col.dtype
        cdef libcudf_types.type_id tid

        if is_list_dtype(self.dtype):
            tid = libcudf_types.type_id.LIST
        elif is_struct_dtype(self.dtype):
            tid = libcudf_types.type_id.STRUCT
        else:
            tid = <libcudf_types.type_id> (
                <underlying_type_t_type_id> (
                    np_to_cudf_types[np.dtype(data_dtype)]
                )
            )

        cdef libcudf_types.data_type dtype = libcudf_types.data_type(tid)
        cdef libcudf_types.size_type offset = self.offset
        cdef vector[column_view] children
        cdef void* data

        data = <void*><uintptr_t>(col.base_data_ptr)

        cdef Column child_column
        if col.base_children:
            for child_column in col.base_children:
                children.push_back(child_column.view())

        cdef libcudf_types.bitmask_type* mask
        if self.nullable:
            mask = <libcudf_types.bitmask_type*><uintptr_t>(self.base_mask_ptr)
        else:
            mask = NULL

        cdef libcudf_types.size_type c_null_count = null_count

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
        cdef column_view view = c_col.get()[0].view()
        cdef libcudf_types.type_id tid = view.type().id()
        cdef libcudf_types.data_type c_dtype
        cdef size_type length = view.size()
        cdef libcudf_types.mask_state mask_state
        if tid == libcudf_types.type_id.TIMESTAMP_DAYS:
            c_dtype = libcudf_types.data_type(
                libcudf_types.type_id.TIMESTAMP_SECONDS
            )
            with nogil:
                c_col = move(libcudf_unary.cast(view, c_dtype))
        elif tid == libcudf_types.type_id.EMPTY:
            c_dtype = libcudf_types.data_type(libcudf_types.type_id.INT8)
            mask_state = libcudf_types.mask_state.ALL_NULL
            with nogil:
                c_col = move(make_numeric_column(c_dtype, length, mask_state))

        size = c_col.get()[0].size()
        dtype = dtype_from_column_view(c_col.get()[0].view())
        has_nulls = c_col.get()[0].has_nulls()

        # After call to release(), c_col is unusable
        cdef column_contents contents = move(c_col.get()[0].release())

        data = DeviceBuffer.c_from_unique_ptr(move(contents.data))
        data = Buffer(data)

        if has_nulls:
            mask = DeviceBuffer.c_from_unique_ptr(move(contents.null_mask))
            mask = Buffer(mask)
            null_count = c_col.get()[0].null_count()
        else:
            mask = None
            null_count = 0

        cdef vector[unique_ptr[column]] c_children = move(contents.children)
        children = ()
        if c_children.size() != 0:
            children = tuple(Column.from_unique_ptr(move(c_children[i]))
                             for i in range(c_children.size()))

        return cudf.core.column.build_column(
            data,
            dtype=dtype,
            mask=mask,
            size=size,
            null_count=null_count,
            children=children
        )

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
        column_owner = isinstance(owner, Column)
        mask_owner = owner
        if column_owner and is_categorical_dtype(owner.dtype):
            owner = owner.base_children[0]

        size = cv.size()
        offset = cv.offset()
        dtype = dtype_from_column_view(cv)

        data_ptr = <uintptr_t>(cv.head[void]())
        data = None
        base_size = size + offset
        data_owner = owner
        if column_owner:
            data_owner = owner.base_data
            base_size = owner.base_size
        if data_ptr:
            if data_owner is None:
                data = Buffer(
                    rmm.DeviceBuffer(ptr=data_ptr,
                                     size=(size+offset) * dtype.itemsize)
                )
            else:
                data = Buffer(
                    data=data_ptr,
                    size=(base_size) * dtype.itemsize,
                    owner=data_owner
                )
        else:
            data = Buffer(
                rmm.DeviceBuffer(ptr=data_ptr, size=0)
            )

        mask_ptr = <uintptr_t>(cv.null_mask())
        mask = None
        if mask_ptr:
            if column_owner:
                mask_owner = mask_owner.base_mask
            if mask_owner is None:
                mask = Buffer(
                    rmm.DeviceBuffer(
                        ptr=mask_ptr,
                        size=bitmask_allocation_size_bytes(size+offset)
                    )
                )
            else:
                mask = Buffer(
                    data=mask_ptr,
                    size=bitmask_allocation_size_bytes(base_size),
                    owner=mask_owner
                )

        if cv.has_nulls():
            null_count = cv.null_count()
        else:
            null_count = 0

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

        result = cudf.core.column.build_column(
            data,
            dtype,
            mask,
            size,
            offset,
            null_count,
            tuple(children)
        )

        return result


def make_column_from_scalar(Scalar val, size_type size):
    cdef scalar* c_val = val.c_value.get()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_make_column_from_scalar(c_val[0], size))

    return Column.from_unique_ptr(move(c_result))
