# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport (
    uintptr_t,
    uint64_t,
    uint32_t,
    uint16_t,
    uint8_t,
    int64_t,
    int32_t,
    int16_t,
    int8_t,
)

from cpython.ref cimport PyObject

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.stream cimport Stream
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport bitmask_type, size_type
from pylibcudf.libcudf.interop cimport ArrowArray

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType
from .scalar cimport Scalar


cdef extern from "Python.h":
    # Need the C functions/macros that return PyObject*
    # otherwise we'd get errors like "Cannot convert Python object to 'PyObject *'"
    ctypedef Py_ssize_t Py_ssize_t
    cdef PyObject* PyList_New(Py_ssize_t size)
    void PyList_SET_ITEM(
        PyObject* list,
        Py_ssize_t index,
        PyObject* item
    )
    cdef PyObject* PyLong_FromLongLong(long long val)
    cdef PyObject* PyLong_FromLong(long val)
    cdef PyObject* PyLong_FromUnsignedLong(unsigned long val)
    cdef PyObject* PyLong_FromUnsignedLongLong(
        unsigned long long val
    )
    cdef PyObject* PyFloat_FromDouble(double val)
    cdef PyObject* PyUnicode_DecodeUTF8(
        const char* s,
        Py_ssize_t size,
        const char* errors
    )
    void Py_INCREF(PyObject* obj)
    cdef PyObject* Py_None
    cdef PyObject* Py_True
    cdef PyObject* Py_False


cdef class OwnerWithCAI:
    cdef object owner
    cdef dict cai

    @staticmethod
    cdef create(column_view cv, object owner)


cdef class OwnerMaskWithCAI:
    cdef object owner
    cdef dict cai

    @staticmethod
    cdef create(column_view cv, object owner)


cdef gpumemoryview _copy_array_to_device(object buf)


cdef class Column:
    # TODO: Should we document these attributes? Should we mark them readonly?
    cdef:
        # Core data
        DataType _data_type
        size_type _size
        gpumemoryview _data
        gpumemoryview _mask
        size_type _null_count
        size_type _offset
        # _children: List[Column]
        list _children
        size_type _num_children

    cdef column_view view(self) nogil
    cdef mutable_column_view mutable_view(self) nogil

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col, Stream stream=*)

    @staticmethod
    cdef Column from_column_view(const column_view& cv, Column owner)

    @staticmethod
    cdef Column from_column_view_of_arbitrary(const column_view& cv, object owner)

    @staticmethod
    cdef Column _wrap_nested_list_column(
        gpumemoryview data,
        tuple shape,
        DataType dtype,
        Column base=*,
    )

    cpdef Scalar to_scalar(self)
    cpdef list to_pylist_slow(self, Stream stream = *)
    cpdef list to_pylist(self)
    cpdef DataType type(self)
    cpdef Column child(self, size_type index)
    cpdef size_type num_children(self)
    cpdef size_type size(self)
    cpdef size_type null_count(self)
    cpdef size_type offset(self)
    cpdef gpumemoryview data(self)
    cpdef gpumemoryview null_mask(self)
    cpdef list children(self)
    cpdef Column copy(self)
    cpdef uint64_t device_buffer_size(self)
    cpdef Column with_mask(self, gpumemoryview, size_type)

    cpdef ListColumnView list_view(self)


cdef class ListColumnView:
    """Accessor for methods of a Column that are specific to lists."""
    cdef Column _column
    cpdef child(self)
    cpdef offsets(self)
    cdef lists_column_view view(self) nogil
