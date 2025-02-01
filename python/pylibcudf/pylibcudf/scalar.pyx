# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cpython cimport bool as py_bool
from cpython.datetime cimport datetime
from cython cimport no_gc_clear
from libc.stdint cimport int64_t
from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.scalar.scalar cimport (
    scalar,
    numeric_scalar,
    timestamp_scalar,
)
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_empty_scalar_like,
    make_numeric_scalar,
    make_string_scalar,
    make_timestamp_scalar,
)
from pylibcudf.libcudf.wrappers.timestamps cimport timestamp_us
from pylibcudf.libcudf.types cimport type_id


from rmm.pylibrmm.memory_resource cimport get_current_device_resource

from .column cimport Column
from .types cimport DataType

__all__ = ["Scalar"]


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the Scalar is in a reference cycle. Removing the tp_clear
# function with the no_gc_clear decoration prevents that. See
# https://github.com/rapidsai/rmm/pull/931 for details.
@no_gc_clear
cdef class Scalar:
    """A scalar value in device memory.

    This is the Cython representation of :cpp:class:`cudf::scalar`.
    """
    # Unlike for columns, libcudf does not support scalar views. All APIs that
    # accept scalar values accept references to the owning object rather than a
    # special view type. As a result, pylibcudf.Scalar has a simpler structure
    # than pylibcudf.Column because it can be a true wrapper around a libcudf
    # column

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

    def __init__(self, *args, **kwargs):
        # TODO: This case is not something we really want to
        # support, but it here for now to ease the transition of
        # DeviceScalar.
        raise ValueError("Scalar should be constructed with a factory")

    __hash__ = None

    cdef const scalar* get(self) noexcept nogil:
        return self.c_obj.get()

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef bool is_valid(self):
        """True if the scalar is valid, false if not"""
        return self.get().is_valid()

    @staticmethod
    cdef Scalar empty_like(Column column):
        """Construct a null scalar with the same type as column.

        Parameters
        ----------
        column
            Column to take type from

        Returns
        -------
        New empty (null) scalar of the given type.
        """
        return Scalar.from_libcudf(move(make_empty_scalar_like(column.view())))

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=None):
        """Construct a Scalar object from a libcudf scalar.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_obj.swap(libcudf_scalar)
        s._data_type = DataType.from_libcudf(s.get().type())
        return s

    @classmethod
    def from_py(cls, py_val):
        """Convert a Python standard library object to a Scalar.
        """
        cdef DataType dtype
        if isinstance(py_val, py_bool):
            dtype = DataType(type_id.BOOL8)
            c_val = make_numeric_scalar(dtype.c_obj)
            (<numeric_scalar[cbool]*>c_val.get()).set_value(py_val)
        elif isinstance(py_val, int):
            dtype = DataType(type_id.INT64)
            c_val = make_numeric_scalar(dtype.c_obj)
            (<numeric_scalar[int64_t]*>c_val.get()).set_value(py_val)
        elif isinstance(py_val, float):
            dtype = DataType(type_id.FLOAT64)
            c_val = make_numeric_scalar(dtype.c_obj)
            (<numeric_scalar[double]*>c_val.get()).set_value(py_val)
        elif isinstance(py_val, str):
            dtype = DataType(type_id.STRING)
            c_val = make_string_scalar(py_val.encode())
        #elif isinstance(py_val, datetime.datetime):
        #    if py_val.microsecond != 0:
        #        raise NotImplementedError("Non-zero microseconds is not supported.")
        #    if py_val.tzinfo is not None:
        #        raise NotImplementedError(f"{py_val.tzinfo=} is not supported.")
        #    dtype = DataType(type_id.TIMESTAMP_MICROSECONDS)
        #    c_val = timestamp_scalar(<int64_t>int(py_val.timestamp()))
            #c_val = make_timestamp_scalar(dtype.c_obj)
            #(<timestamp_scalar[timestamp_us]*>c_val.get()).set_value(py_val)
        else:
            raise NotImplementedError(f"{type(py_val).__name__} is not supported.")

        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_obj.swap(c_val)
        s._data_type = dtype
        return s
