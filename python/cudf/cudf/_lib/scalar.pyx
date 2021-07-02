# Copyright (c) 2020, NVIDIA CORPORATION.
import decimal
import numpy as np
import pandas as pd
import pyarrow as pa
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
from cudf.core.dtypes import ListDtype, StructDtype
from cudf._lib.types import (
    cudf_to_np_types,
    duration_unit_map
)
from cudf._lib.types import datetime_unit_map
from cudf._lib.types cimport underlying_type_t_type_id, dtype_from_column_view

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.table cimport Table
from cudf._lib.interop import to_arrow

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
from cudf._lib.cpp.wrappers.decimals cimport decimal64, scale_type
from cudf._lib.cpp.scalar.scalar cimport (
    scalar,
    numeric_scalar,
    timestamp_scalar,
    duration_scalar,
    string_scalar,
    fixed_point_scalar,
    list_scalar,
    struct_scalar
)
from cudf.utils.dtypes import _decimal_to_int64, is_list_dtype, is_struct_dtype
cimport cudf._lib.cpp.types as libcudf_types

cdef class DeviceScalar:

    def __init__(self, value, dtype):
        """
        Type representing an *immutable* scalar value on the device

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
        self._dtype = dtype if dtype.kind != 'U' else np.dtype('object')
        self._set_value(value, self._dtype)

    def _set_value(self, value, dtype):
        # IMPORTANT: this should only ever be called from __init__
        valid = not _is_null_host_scalar(value)

        if isinstance(dtype, cudf.Decimal64Dtype):
            _set_decimal64_from_scalar(
                self.c_value, value, dtype, valid)
        elif isinstance(dtype, cudf.ListDtype):
            _set_list_from_pylist(
                self.c_value, value, dtype, valid)
        elif pd.api.types.is_string_dtype(dtype):
            _set_string_from_np_string(self.c_value, value, valid)
        elif pd.api.types.is_numeric_dtype(dtype):
            _set_numeric_from_np_scalar(self.c_value,
                                        value,
                                        dtype,
                                        valid)
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

    def _to_host_scalar(self):
        if isinstance(self.dtype, cudf.Decimal64Dtype):
            result = _get_py_decimal_from_fixed_point(self.c_value)
        elif is_struct_dtype(self.dtype):
            result = _get_py_dict_from_struct(self.c_value)
        elif is_list_dtype(self.dtype):
            result = _get_py_list_from_list(self.c_value)
        elif pd.api.types.is_string_dtype(self.dtype):
            result = _get_py_string_from_string(self.c_value)
        elif pd.api.types.is_numeric_dtype(self.dtype):
            result = _get_np_scalar_from_numeric(self.c_value)
        elif pd.api.types.is_datetime64_dtype(self.dtype):
            result = _get_np_scalar_from_timestamp64(self.c_value)
        elif pd.api.types.is_timedelta64_dtype(self.dtype):
            result = _get_np_scalar_from_timedelta64(self.c_value)
        else:
            raise ValueError(
                "Could not convert cudf::scalar to a Python value"
            )
        return result

    @property
    def dtype(self):
        """
        The NumPy dtype corresponding to the data type of the underlying
        device scalar.
        """
        return self._dtype

    @property
    def value(self):
        """
        Returns a host copy of the underlying device scalar.
        """
        return self._to_host_scalar()

    cdef const scalar* get_raw_ptr(self) except *:
        return self.c_value.get()

    cpdef bool is_valid(self):
        """
        Returns if the Scalar is valid or not(i.e., <NA>).
        """
        return self.get_raw_ptr()[0].is_valid()

    def __repr__(self):
        if self.value is cudf.NA:
            return (
                f"{self.__class__.__name__}"
                f"({self.value}, {self.dtype.__repr__()})"
            )
        else:
            return f"{self.__class__.__name__}({self.value.__repr__()})"

    @staticmethod
    cdef DeviceScalar from_unique_ptr(unique_ptr[scalar] ptr, dtype=None):
        """
        Construct a Scalar object from a unique_ptr<cudf::scalar>.
        """
        cdef DeviceScalar s = DeviceScalar.__new__(DeviceScalar)
        cdef libcudf_types.data_type cdtype

        s.c_value = move(ptr)
        cdtype = s.get_raw_ptr()[0].type()

        if cdtype.id() == libcudf_types.DECIMAL64 and dtype is None:
            raise TypeError(
                "Must pass a dtype when constructing from a fixed-point scalar"
            )
        elif cdtype.id() == libcudf_types.STRUCT:
            struct_table_view = (<struct_scalar*>s.get_raw_ptr())[0].view()
            s._dtype = StructDtype({
                str(i): dtype_from_column_view(struct_table_view.column(i))
                for i in range(struct_table_view.num_columns())
            })
        elif cdtype.id() == libcudf_types.LIST:
            if (
                <list_scalar*>s.get_raw_ptr()
            )[0].view().type().id() == libcudf_types.LIST:
                s._dtype = dtype_from_column_view(
                    (<list_scalar*>s.get_raw_ptr())[0].view()
                )
            else:
                s._dtype = ListDtype(
                    cudf_to_np_types[
                        <underlying_type_t_type_id>(
                            (<list_scalar*>s.get_raw_ptr())[0]
                            .view().type().id()
                        )
                    ]
                )
        else:
            if dtype is not None:
                s._dtype = dtype
            else:
                s._dtype = cudf_to_np_types[
                    <underlying_type_t_type_id>(cdtype.id())
                ]
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

cdef _set_decimal64_from_scalar(unique_ptr[scalar]& s,
                                object value,
                                object dtype,
                                bool valid=True):
    value = _decimal_to_int64(value) if valid else 0
    s.reset(
        new fixed_point_scalar[decimal64](
            <int64_t>np.int64(value), scale_type(-dtype.scale), valid
        )
    )

cdef _get_py_dict_from_struct(unique_ptr[scalar]& s):
    if not s.get()[0].is_valid():
        return cudf.NA

    cdef table_view struct_table_view = (<struct_scalar*>s.get()).view()
    columns = [str(i) for i in range(struct_table_view.num_columns())]

    cdef Table to_arrow_table = Table.from_table_view(
        struct_table_view,
        None,
        column_names=columns
    )

    python_dict = to_arrow(to_arrow_table, columns).to_pydict()

    return {k: _nested_na_replace(python_dict[k])[0] for k in python_dict}

cdef _set_list_from_pylist(unique_ptr[scalar]& s,
                           object value,
                           object dtype,
                           bool valid=True):

    value = value if valid else [cudf.NA]
    cdef Column col
    if isinstance(dtype.element_type, ListDtype):
        pa_type = dtype.element_type.to_arrow()
    else:
        pa_type = dtype.to_arrow().value_type
    col = cudf.core.column.as_column(
        pa.array(value, from_pandas=True, type=pa_type)
    )
    cdef column_view col_view = col.view()
    s.reset(
        new list_scalar(col_view, valid)
    )


cdef _get_py_list_from_list(unique_ptr[scalar]& s):

    if not s.get()[0].is_valid():
        return cudf.NA

    cdef column_view list_col_view = (<list_scalar*>s.get()).view()
    cdef Column list_col = Column.from_column_view(list_col_view, None)
    cdef Table to_arrow_table = Table({"col": list_col})

    arrow_table = to_arrow(to_arrow_table, [["col", []]])
    result = arrow_table['col'].to_pylist()
    return _nested_na_replace(result)


cdef _get_py_string_from_string(unique_ptr[scalar]& s):
    if not s.get()[0].is_valid():
        return cudf.NA
    return (<string_scalar*>s.get())[0].to_string().decode()


cdef _get_np_scalar_from_numeric(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return cudf.NA

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


cdef _get_py_decimal_from_fixed_point(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return cudf.NA

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.DECIMAL64:
        rep_val = int((<fixed_point_scalar[decimal64]*>s_ptr)[0].value())
        scale = int((<fixed_point_scalar[decimal64]*>s_ptr)[0].type().scale())
        return decimal.Decimal(rep_val).scaleb(scale)
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")

cdef _get_np_scalar_from_timestamp64(unique_ptr[scalar]& s):

    cdef scalar* s_ptr = s.get()

    if not s_ptr[0].is_valid():
        return cudf.NA

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


def as_device_scalar(val, dtype=None):
    if isinstance(val, (cudf.Scalar, DeviceScalar)):
        if dtype == val.dtype or dtype is None:
            if isinstance(val, DeviceScalar):
                return val
            else:
                return val.device_value
        else:
            raise TypeError("Can't update dtype of existing GPU scalar")
    else:
        return cudf.Scalar(val, dtype=dtype).device_value


def _is_null_host_scalar(slr):
    if slr is None or slr is cudf.NA:
        return True
    elif isinstance(slr, (np.datetime64, np.timedelta64)) and np.isnat(slr):
        return True
    else:
        return False


def _create_proxy_nat_scalar(dtype):
    cdef DeviceScalar result = DeviceScalar.__new__(DeviceScalar)

    dtype = np.dtype(dtype)
    if dtype.char in 'mM':
        nat = dtype.type('NaT').astype(dtype)
        if dtype.type == np.datetime64:
            _set_datetime64_from_np_scalar(result.c_value, nat, dtype, True)
        elif dtype.type == np.timedelta64:
            _set_timedelta64_from_np_scalar(result.c_value, nat, dtype, True)
        return result
    else:
        raise TypeError('NAT only valid for datetime and timedelta')


def _nested_na_replace(input_list):
    '''
    Replace `None` with `cudf.NA` in the result of
    `__getitem__` calls to list type columns
    '''
    for idx, value in enumerate(input_list):
        if isinstance(value, list):
            _nested_na_replace(value)
        elif value is None:
            input_list[idx] = cudf.NA
    return input_list
