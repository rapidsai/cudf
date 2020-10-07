# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp cimport bool

import cudf
from cudf._lib.types import cudf_to_np_types, duration_unit_map
from cudf._lib.types import datetime_unit_map
from cudf._lib.types cimport underlying_type_t_type_id

from cudf._lib.cpp.wrappers.timestamps cimport (
    timestamp_s,
    timestamp_ms,
    timestamp_us,
    timestamp_ns
)
from cudf._lib.cpp.wrappers.durations cimport(
    duration_s,
    duration_ms,
    duration_us,
    duration_ns
)
from cudf._lib.cpp.scalar.scalar cimport (
    scalar,
    numeric_scalar,
    timestamp_scalar,
    duration_scalar,
    string_scalar
)
cimport cudf._lib.cpp.types as libcudf_types


cdef class Scalar:

    def __init__(self, value, dtype=None):
        """
        cudf.Scalar: Type representing a scalar value on the device

        Parameters
        ----------
        value : scalar
            An object of scalar type, i.e., one for which
            `np.isscalar()` returns `True`. Can also be `None`,
            to represent a "null" scalar. In this case,
            dtype *must* be provided.
        dtype : dtype
            A NumPy dtype.
        """

        value = cudf.utils.dtypes.to_cudf_compatible_scalar(value, dtype=dtype)

        valid = value is not None

        if dtype is None:
            if value is None:
                raise TypeError(
                    "dtype required when constructing a null scalar"
                )
            else:
                dtype = value.dtype

        dtype = np.dtype(dtype)

        if pd.api.types.is_string_dtype(dtype):
            _set_string_from_np_string(self.c_value, value, valid)
        elif pd.api.types.is_numeric_dtype(dtype):
            _set_numeric_from_np_scalar(self.c_value, value, dtype, valid)
        elif pd.api.types.is_datetime64_dtype(dtype):
            _set_datetime64_from_np_scalar(
                self.c_value, value, dtype, valid
            )
        elif pd.api.types.is_timedelta64_dtype(dtype):
            _set_timedelta64_from_np_scalar(
                self.c_value, value, dtype, valid
            )
        else:
            raise ValueError(
                f"Cannot convert value of type "
                f"{type(value).__name__} to cudf scalar"
            )

    @property
    def dtype(self):
        """
        The NumPy dtype corresponding to the data type of the underlying
        device scalar.
        """
        cdef libcudf_types.data_type cdtype = self.c_value.get()[0].type()
        return cudf_to_np_types[<underlying_type_t_type_id>(cdtype.id())]

    @property
    def value(self):
        """
        Returns a host copy of the underlying device scalar.
        """
        if pd.api.types.is_string_dtype(self.dtype):
            return _get_py_string_from_string(self.c_value)
        elif pd.api.types.is_numeric_dtype(self.dtype):
            return _get_np_scalar_from_numeric(self.c_value)
        elif pd.api.types.is_datetime64_dtype(self.dtype):
            return _get_np_scalar_from_timestamp64(self.c_value)
        elif pd.api.types.is_timedelta64_dtype(self.dtype):
            return _get_np_scalar_from_timedelta64(self.c_value)
        else:
            raise ValueError(
                "Could not convert cudf::scalar to a Python value"
            )

    cpdef bool is_valid(self):
        """
        Returns if the Scalar is valid or not(i.e., <NA>).
        """
        return self.c_value.get()[0].is_valid()

    def __repr__(self):
        if self.value is None:
            return f"Scalar({self.value}, {self.dtype.__repr__()})"
        else:
            return f"Scalar({self.value.__repr__()})"

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr):
        """
        Construct a Scalar object from a unique_ptr<cudf::scalar>.
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_value = move(ptr)
        return s


cdef _set_string_from_np_string(unique_ptr[scalar]& s, value, bool valid=True):
    value = value if valid else ""
    s.reset(new string_scalar(value.encode(), valid))


cdef _set_numeric_from_np_scalar(unique_ptr[scalar]& s,
                                 object value,
                                 object dtype,
                                 bool valid=True):
    value = value if valid else 0
    if dtype == "int8":
        s.reset(new numeric_scalar[int8_t](value, valid))
    elif dtype == "int16":
        s.reset(new numeric_scalar[int16_t](value, valid))
    elif dtype == "int32":
        s.reset(new numeric_scalar[int32_t](value, valid))
    elif dtype == "int64":
        s.reset(new numeric_scalar[int64_t](value, valid))
    elif dtype == "uint8":
        s.reset(new numeric_scalar[uint8_t](value, valid))
    elif dtype == "uint16":
        s.reset(new numeric_scalar[uint16_t](value, valid))
    elif dtype == "uint32":
        s.reset(new numeric_scalar[uint32_t](value, valid))
    elif dtype == "uint64":
        s.reset(new numeric_scalar[uint64_t](value, valid))
    elif dtype == "float32":
        s.reset(new numeric_scalar[float](value, valid))
    elif dtype == "float64":
        s.reset(new numeric_scalar[double](value, valid))
    elif dtype == "bool":
        s.reset(new numeric_scalar[bool](<bool>value, valid))
    else:
        raise ValueError(f"dtype not supported: {dtype}")


cdef _set_datetime64_from_np_scalar(unique_ptr[scalar]& s,
                                    object value,
                                    object dtype,
                                    bool valid=True):

    value = value if valid else 0

    if dtype == "datetime64[s]":
        s.reset(
            new timestamp_scalar[timestamp_s](<int64_t>np.int64(value), valid)
        )
    elif dtype == "datetime64[ms]":
        s.reset(
            new timestamp_scalar[timestamp_ms](<int64_t>np.int64(value), valid)
        )
    elif dtype == "datetime64[us]":
        s.reset(
            new timestamp_scalar[timestamp_us](<int64_t>np.int64(value), valid)
        )
    elif dtype == "datetime64[ns]":
        s.reset(
            new timestamp_scalar[timestamp_ns](<int64_t>np.int64(value), valid)
        )
    else:
        raise ValueError(f"dtype not supported: {dtype}")

cdef _set_timedelta64_from_np_scalar(unique_ptr[scalar]& s,
                                     object value,
                                     object dtype,
                                     bool valid=True):

    value = value if valid else 0

    if dtype == "timedelta64[s]":
        s.reset(
            new duration_scalar[duration_s](<int64_t>np.int64(value), valid)
        )
    elif dtype == "timedelta64[ms]":
        s.reset(
            new duration_scalar[duration_ms](<int64_t>np.int64(value), valid)
        )
    elif dtype == "timedelta64[us]":
        s.reset(
            new duration_scalar[duration_us](<int64_t>np.int64(value), valid)
        )
    elif dtype == "timedelta64[ns]":
        s.reset(
            new duration_scalar[duration_ns](<int64_t>np.int64(value), valid)
        )
    else:
        raise ValueError(f"dtype not supported: {dtype}")

cdef _get_py_string_from_string(unique_ptr[scalar]& s):
    if not s.get()[0].is_valid():
        return None
    return (<string_scalar*>s.get())[0].to_string().decode()


cdef _get_np_scalar_from_numeric(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.INT8:
        return np.int8((<numeric_scalar[int8_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.INT16:
        return np.int16((<numeric_scalar[int16_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.INT32:
        return np.int32((<numeric_scalar[int32_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.INT64:
        return np.int64((<numeric_scalar[int64_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT8:
        return np.uint8((<numeric_scalar[uint8_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT16:
        return np.uint16((<numeric_scalar[uint16_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT32:
        return np.uint32((<numeric_scalar[uint32_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT64:
        return np.uint64((<numeric_scalar[uint64_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.FLOAT32:
        return np.float32((<numeric_scalar[float]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.FLOAT64:
        return np.float64((<numeric_scalar[double]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.BOOL8:
        return np.bool_((<numeric_scalar[bool]*>s_ptr)[0].value())
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


cdef _get_np_scalar_from_timestamp64(unique_ptr[scalar]& s):

    cdef scalar* s_ptr = s.get()

    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.TIMESTAMP_SECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MILLISECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MICROSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_NANOSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


cdef _get_np_scalar_from_timedelta64(unique_ptr[scalar]& s):

    cdef scalar* s_ptr = s.get()

    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.DURATION_SECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_s]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.DURATION_MILLISECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_ms]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.DURATION_MICROSECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_us]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.DURATION_NANOSECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_ns]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


def as_scalar(val, dtype=None):
    if isinstance(val, Scalar):
        if (dtype is None or dtype == val.dtype):
            return val
        else:
            return Scalar(val.value, dtype)
    else:
        return Scalar(value=val, dtype=dtype)
