# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cpython cimport bool as py_bool, datetime
from cython cimport no_gc_clear
from libc.stdint cimport int64_t
from libc.time cimport tm, mktime, time_t
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
    make_string_scalar,
    make_numeric_scalar,
    make_timestamp_scalar,
)
from pylibcudf.libcudf.types cimport type_id
from pylibcudf.libcudf.wrappers.timestamps cimport (
    timestamp_us,
    nanoseconds,
    microseconds,
    time_point,
    system_clock,
    time_point_cast,
)


from rmm.pylibrmm.memory_resource cimport get_current_device_resource

from .column cimport Column
from .types cimport DataType

from functools import singledispatch

__all__ = ["Scalar"]


cdef time_point[system_clock, nanoseconds] _datetime_to_time_point(
    datetime.datetime dt
):
    if dt.tzinfo is not None:
        raise NotImplementedError("datetimes with timezones are not supported.")
    if dt.microsecond != 0:
        raise NotImplementedError("Non-zero microseconds are not supported.")
    cdef tm time_struct
    time_struct.tm_year = dt.year - 1900
    time_struct.tm_mon = dt.month - 1
    time_struct.tm_mday = dt.day
    time_struct.tm_hour = dt.hour
    time_struct.tm_min = dt.minute
    time_struct.tm_sec = dt.second
    cdef time_t time = mktime(&time_struct)
    cdef time_point[system_clock, nanoseconds] tp = system_clock.from_time_t(time)
    return tp


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
        """
        Convert a Python standard library object to a Scalar.

        Parameters
        ----------
        py_val: bool, int, float, str, datetime.datetime, datetime.timedelta, list, dict
            Value to convert to a pylibcudf.Scalar

        Returns
        -------
        Scalar
            New pylibcudf.Scalar
        """
        return _from_py(py_val)

cdef Scalar _new_scalar(unique_ptr[scalar] c_obj, DataType dtype):
    cdef Scalar s = Scalar.__new__(Scalar)
    s.c_obj.swap(c_obj)
    s._data_type = dtype
    return s


@singledispatch
def _from_py(py_val):
    raise TypeError(f"{type(py_val).__name__} cannot be converted to pylibcudf.Scalar")


@_from_py.register(dict)
@_from_py.register(list)
@_from_py.register(datetime.datetime)
@_from_py.register(datetime.timedelta)
def _(py_val):
    raise NotImplementedError(
        f"Conversion from {type(py_val).__name__} is currently not supported."
    )


@_from_py.register(float)
def _(py_val):
    cdef DataType dtype = DataType(type_id.FLOAT64)
    cdef unique_ptr[scalar] c_obj = make_numeric_scalar(dtype.c_obj)
    (<numeric_scalar[double]*>c_obj.get()).set_value(py_val)
    cdef Scalar slr = _new_scalar(move(c_obj), dtype)
    return slr


@_from_py.register(int)
def _(py_val):
    cdef DataType dtype = DataType(type_id.INT64)
    cdef unique_ptr[scalar] c_obj = make_numeric_scalar(dtype.c_obj)
    (<numeric_scalar[int64_t]*>c_obj.get()).set_value(py_val)
    cdef Scalar slr = _new_scalar(move(c_obj), dtype)
    return slr


@_from_py.register(py_bool)
def _(py_val):
    cdef DataType dtype = DataType(type_id.BOOL8)
    cdef unique_ptr[scalar] c_obj = make_numeric_scalar(dtype.c_obj)
    (<numeric_scalar[cbool]*>c_obj.get()).set_value(py_val)
    cdef Scalar slr = _new_scalar(move(c_obj), dtype)
    return slr


@_from_py.register(str)
def _(py_val):
    cdef DataType dtype = DataType(type_id.STRING)
    cdef unique_ptr[scalar] c_obj = make_string_scalar(py_val.encode())
    cdef Scalar slr = _new_scalar(move(c_obj), dtype)
    return slr


@_from_py.register(datetime.datetime)
def _(py_val):
    cdef DataType dtype = DataType(type_id.TIMESTAMP_MICROSECONDS)
    cdef unique_ptr[scalar] c_obj = make_timestamp_scalar(dtype.c_obj)
    cdef time_point[system_clock, nanoseconds] tp = _datetime_to_time_point(py_val)
    cdef time_point[system_clock, microseconds] tp_casted = (
        time_point_cast[system_clock, microseconds, nanoseconds](tp)
    )
    (<timestamp_scalar[timestamp_us]*>c_obj.get()).set_value(tp_casted)
    cdef Scalar slr = _new_scalar(move(c_obj), dtype)
    return slr
