# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import copy
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
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

import cudf

from cudf._lib.pylibcudf.types cimport type_id

from cudf._lib.types import (
    LIBCUDF_TO_SUPPORTED_NUMPY_TYPES,
    datetime_unit_map,
    duration_unit_map,
)
from cudf.core.dtypes import ListDtype, StructDtype
from cudf.core.missing import NA, NaT

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.types cimport dtype_from_column_view, underlying_type_t_type_id

from cudf._lib.interop import to_arrow

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib cimport pylibcudf
from cudf._lib.cpp.scalar.scalar cimport (
    duration_scalar,
    fixed_point_scalar,
    list_scalar,
    numeric_scalar,
    scalar,
    string_scalar,
    struct_scalar,
    timestamp_scalar,
)
from cudf._lib.cpp.wrappers.decimals cimport decimal32, decimal64, decimal128
from cudf._lib.cpp.wrappers.durations cimport (
    duration_ms,
    duration_ns,
    duration_s,
    duration_us,
)
from cudf._lib.cpp.wrappers.timestamps cimport (
    timestamp_ms,
    timestamp_ns,
    timestamp_s,
    timestamp_us,
)
from cudf._lib.utils cimport columns_from_table_view


# TODO: Check if this could replace _nested_na_replace
def _replace_nested_nulls(obj):
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            # TODO: Check if this should use _is_null_host_scalar
            if cudf.utils.utils.is_na_like(item):
                obj[i] = None
            elif isinstance(item, (dict, list)):
                _replace_nested_nulls(item)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if cudf.utils.utils.is_na_like(v):
                obj[k] = None
            elif isinstance(v, (dict, list)):
                _replace_nested_nulls(v)


cdef class DeviceScalar:

    # I think this should be removable, except that currently the way that
    # from_unique_ptr is implemented is probably dereferencing this in an
    # invalid state. See what the best way to fix that is.
    def __cinit__(self, *args, **kwargs):
        self.c_value = pylibcudf.Scalar()

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
        dtype = dtype if dtype.kind != 'U' else cudf.dtype('object')

        if cudf.utils.utils.is_na_like(value):
            value = None
        else:
            # TODO: For now we always deepcopy the input value to avoid
            # overwriting the input values when replacing nulls. Since it's
            # just host values it's not that expensive, but we could consider
            # alternatives.
            value = copy.deepcopy(value)
            _replace_nested_nulls(value)

        if isinstance(dtype, cudf.core.dtypes._BaseDtype):
            pa_type = dtype.to_arrow()
        elif pd.api.types.is_string_dtype(dtype):
            # Have to manually convert object types, which we use internally
            # for strings but pyarrow only supports as unicode 'U'
            pa_type = pa.string()
        else:
            pa_type = pa.from_numpy_dtype(dtype)

        pa_scalar = pa.scalar(value, type=pa_type)

        cdef type_id tid
        data_type = None
        if isinstance(dtype, cudf.core.dtypes.DecimalDtype):
            tid = type_id.DECIMAL128
            if isinstance(dtype, cudf.core.dtypes.Decimal32Dtype):
                tid = type_id.DECIMAL32
            elif isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
                tid = type_id.DECIMAL64
            data_type = pylibcudf.DataType(tid, -dtype.scale)

        self.c_value = pylibcudf.Scalar.from_pyarrow_scalar(pa_scalar, data_type)
        self._dtype = dtype

    def _to_host_scalar(self):
        if isinstance(self.dtype, cudf.core.dtypes.DecimalDtype):
            result = _get_py_decimal_from_fixed_point(self.c_value.c_obj)
        elif cudf.api.types.is_struct_dtype(self.dtype):
            result = _get_py_dict_from_struct(self.c_value.c_obj, self.dtype)
        elif cudf.api.types.is_list_dtype(self.dtype):
            result = _get_py_list_from_list(self.c_value.c_obj, self.dtype)
        elif pd.api.types.is_string_dtype(self.dtype):
            result = _get_py_string_from_string(self.c_value.c_obj)
        elif pd.api.types.is_numeric_dtype(self.dtype):
            result = _get_np_scalar_from_numeric(self.c_value.c_obj)
        elif pd.api.types.is_datetime64_dtype(self.dtype):
            result = _get_np_scalar_from_timestamp64(self.c_value.c_obj)
        elif pd.api.types.is_timedelta64_dtype(self.dtype):
            result = _get_np_scalar_from_timedelta64(self.c_value.c_obj)
        else:
            raise ValueError(
                "Could not convert cudf::scalar to a Python value"
            )
        return result

    # TODO: This is just here for testing and should be removed.
    def get(self):
        return self.c_value

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
        return self.c_value.c_obj.get()

    cpdef bool is_valid(self):
        """
        Returns if the Scalar is valid or not(i.e., <NA>).
        """
        return self.c_value.is_valid()

    def __repr__(self):
        if cudf.utils.utils.is_na_like(self.value):
            return (
                f"{self.__class__.__name__}"
                f"({self.value}, {repr(self.dtype)})"
            )
        else:
            return f"{self.__class__.__name__}({repr(self.value)})"

    @staticmethod
    cdef DeviceScalar from_unique_ptr(unique_ptr[scalar] ptr, dtype=None):
        """
        Construct a Scalar object from a unique_ptr<cudf::scalar>.
        """
        cdef DeviceScalar s = DeviceScalar.__new__(DeviceScalar)
        cdef libcudf_types.data_type cdtype

        s.c_value = pylibcudf.Scalar.from_libcudf(move(ptr))
        cdtype = s.get_raw_ptr()[0].type()

        if dtype is not None:
            s._dtype = dtype
        elif cdtype.id() in {
            libcudf_types.type_id.DECIMAL32,
            libcudf_types.type_id.DECIMAL64,
            libcudf_types.type_id.DECIMAL128,
        }:
            raise TypeError(
                "Must pass a dtype when constructing from a fixed-point scalar"
            )
        elif cdtype.id() == libcudf_types.type_id.STRUCT:
            struct_table_view = (<struct_scalar*>s.get_raw_ptr())[0].view()
            s._dtype = StructDtype({
                str(i): dtype_from_column_view(struct_table_view.column(i))
                for i in range(struct_table_view.num_columns())
            })
        elif cdtype.id() == libcudf_types.type_id.LIST:
            if (
                <list_scalar*>s.get_raw_ptr()
            )[0].view().type().id() == libcudf_types.type_id.LIST:
                s._dtype = dtype_from_column_view(
                    (<list_scalar*>s.get_raw_ptr())[0].view()
                )
            else:
                s._dtype = ListDtype(
                    LIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
                        <underlying_type_t_type_id>(
                            (<list_scalar*>s.get_raw_ptr())[0]
                            .view().type().id()
                        )
                    ]
                )
        else:
            s._dtype = LIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
                <underlying_type_t_type_id>(cdtype.id())
            ]
        return s


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

cdef _get_py_dict_from_struct(unique_ptr[scalar]& s, dtype):
    if not s.get()[0].is_valid():
        return NA

    cdef table_view struct_table_view = (<struct_scalar*>s.get()).view()
    columns = columns_from_table_view(struct_table_view, None)
    struct_col = cudf.core.column.build_struct_column(
        names=dtype.fields.keys(),
        children=tuple(columns),
        size=1,
    )
    table = to_arrow([struct_col], [("None", dtype)])
    python_dict = table.to_pydict()["None"][0]
    return {k: _nested_na_replace([python_dict[k]])[0] for k in python_dict}


cdef _get_py_list_from_list(unique_ptr[scalar]& s, dtype):

    if not s.get()[0].is_valid():
        return NA

    cdef column_view list_col_view = (<list_scalar*>s.get()).view()
    cdef Column element_col = Column.from_column_view(list_col_view, None)

    arrow_obj = to_arrow([element_col], [("None", dtype.element_type)])["None"]

    result = arrow_obj.to_pylist()
    return _nested_na_replace(result)


cdef _get_py_string_from_string(unique_ptr[scalar]& s):
    if not s.get()[0].is_valid():
        return NA
    return (<string_scalar*>s.get())[0].to_string().decode()


cdef _get_np_scalar_from_numeric(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return NA

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.type_id.INT8:
        return np.int8((<numeric_scalar[int8_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.INT16:
        return np.int16((<numeric_scalar[int16_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.INT32:
        return np.int32((<numeric_scalar[int32_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.INT64:
        return np.int64((<numeric_scalar[int64_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.UINT8:
        return np.uint8((<numeric_scalar[uint8_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.UINT16:
        return np.uint16((<numeric_scalar[uint16_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.UINT32:
        return np.uint32((<numeric_scalar[uint32_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.UINT64:
        return np.uint64((<numeric_scalar[uint64_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.FLOAT32:
        return np.float32((<numeric_scalar[float]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.FLOAT64:
        return np.float64((<numeric_scalar[double]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.type_id.BOOL8:
        return np.bool_((<numeric_scalar[bool]*>s_ptr)[0].value())
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


cdef _get_py_decimal_from_fixed_point(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return NA

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.type_id.DECIMAL64:
        rep_val = int((<fixed_point_scalar[decimal64]*>s_ptr)[0].value())
        scale = int((<fixed_point_scalar[decimal64]*>s_ptr)[0].type().scale())
        return decimal.Decimal(rep_val).scaleb(scale)
    elif cdtype.id() == libcudf_types.type_id.DECIMAL32:
        rep_val = int((<fixed_point_scalar[decimal32]*>s_ptr)[0].value())
        scale = int((<fixed_point_scalar[decimal32]*>s_ptr)[0].type().scale())
        return decimal.Decimal(rep_val).scaleb(scale)
    elif cdtype.id() == libcudf_types.type_id.DECIMAL128:
        rep_val = int((<fixed_point_scalar[decimal128]*>s_ptr)[0].value())
        scale = int((<fixed_point_scalar[decimal128]*>s_ptr)[0].type().scale())
        return decimal.Decimal(rep_val).scaleb(scale)
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")

cdef _get_np_scalar_from_timestamp64(unique_ptr[scalar]& s):

    cdef scalar* s_ptr = s.get()

    if not s_ptr[0].is_valid():
        return NaT

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.type_id.TIMESTAMP_SECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.type_id.TIMESTAMP_MILLISECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.type_id.TIMESTAMP_MICROSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.type_id.TIMESTAMP_NANOSECONDS:
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
        return NaT

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.type_id.DURATION_SECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_s]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.type_id.DURATION_MILLISECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_ms]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.type_id.DURATION_MICROSECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_us]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.type_id.DURATION_NANOSECONDS:
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
    if cudf.utils.utils.is_na_like(slr):
        return True
    elif isinstance(slr, (np.datetime64, np.timedelta64)) and np.isnat(slr):
        return True
    else:
        return False


def _create_proxy_nat_scalar(dtype):
    cdef DeviceScalar result = DeviceScalar.__new__(DeviceScalar)

    dtype = cudf.dtype(dtype)
    if dtype.char in 'mM':
        nat = dtype.type('NaT').astype(dtype)
        if dtype.type == np.datetime64:
            _set_datetime64_from_np_scalar(result.c_value.c_obj, nat, dtype, True)
        elif dtype.type == np.timedelta64:
            _set_timedelta64_from_np_scalar(result.c_value.c_obj, nat, dtype, True)
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
            input_list[idx] = NA
    return input_list
