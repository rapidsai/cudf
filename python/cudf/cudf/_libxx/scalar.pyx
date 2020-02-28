# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libcpp.memory cimport unique_ptr
from libcpp cimport bool

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types
from cudf._libxx.move cimport move

from cudf._libxx.cpp.wrappers.bool cimport bool8
from cudf._libxx.cpp.wrappers.timestamps cimport (
    timestamp_s,
    timestamp_ms,
    timestamp_us,
    timestamp_ns
)
from cudf._libxx.cpp.scalar.scalar cimport (
    scalar,
    numeric_scalar,
    timestamp_scalar,
    string_scalar
)
cimport cudf._libxx.cpp.types as libcudf_types


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
        from cudf.utils.dtypes import to_cudf_compatible_scalar

        value = to_cudf_compatible_scalar(value, dtype=dtype)
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
        else:
            raise ValueError(
                "Cannot convert value of type {} to cudf scalar".format(
                    type(value).__name__
                )
            )

    @property
    def dtype(self):
        """
        The NumPy dtype corresponding to the data type of the underlying
        device scalar.
        """
        cdef libcudf_types.data_type cdtype = self.c_value.get()[0].type()
        return cudf_to_np_types[cdtype.id()]

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
        else:
            raise ValueError(
                "Could not convert cudf::scalar to a Python value"
            )

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
    value = value.item() if valid else ""
    s.reset(new string_scalar(value.encode(), valid))


cdef _set_numeric_from_np_scalar(unique_ptr[scalar]& s,
                                 value,
                                 dtype,
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
    elif dtype == "float32":
        s.reset(new numeric_scalar[float](value, valid))
    elif dtype == "float64":
        s.reset(new numeric_scalar[double](value, valid))
    elif dtype == "bool":
        s.reset(new numeric_scalar[bool8](<bool>value, valid))
    else:
        raise ValueError("dtype not supported: {}".format(dtype))


cdef _set_datetime64_from_np_scalar(unique_ptr[scalar]& s,
                                    value,
                                    dtype,
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
        raise ValueError("dtype not supported: {}".format(dtype))


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
    elif cdtype.id() == libcudf_types.FLOAT32:
        return np.float32((<numeric_scalar[float]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.FLOAT64:
        return np.float64((<numeric_scalar[double]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.BOOL8:
        return np.bool((<numeric_scalar[bool8]*>s_ptr)[0].value())
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
            "s"
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MILLISECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            "ms"
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MICROSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            "us"
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_NANOSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            "ns"
        )
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")
