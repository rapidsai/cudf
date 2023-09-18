# Copyright (c) 2020-2023, NVIDIA CORPORATION.

cimport cython

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

from rmm._lib.memory_resource cimport get_current_device_resource

import cudf
from cudf._lib.types import LIBCUDF_TO_SUPPORTED_NUMPY_TYPES
from cudf.core.dtypes import ListDtype, StructDtype
from cudf.core.missing import NA, NaT

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.types cimport dtype_from_column_view, underlying_type_t_type_id

from cudf._lib.interop import from_arrow, to_arrow_scalar

cimport cudf._lib.cpp.types as libcudf_types
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
from cudf._lib.cpp.wrappers.decimals cimport (
    decimal32,
    decimal64,
    decimal128,
    scale_type,
)
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
from cudf._lib.utils cimport table_view_from_columns


def _replace_nested_none(obj):
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if item is None:
                obj[i] = NA
            elif isinstance(item, (dict, list)):
                _replace_nested_none(item)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                obj[k] = NA
            elif isinstance(v, (dict, list)):
                _replace_nested_none(v)


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the DeviceScalar is in a reference cycle. Removing
# the tp_clear function with the no_gc_clear decoration prevents that.
# See https://github.com/rapidsai/rmm/pull/931 for details.
@cython.no_gc_clear
cdef class DeviceScalar:

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

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
        self._dtype = dtype if dtype.kind != 'U' else cudf.dtype('object')
        self._set_value(value, self._dtype)

    def _set_value(self, value, dtype):
        # IMPORTANT: this should only ever be called from __init__
        valid = not _is_null_host_scalar(value)

        if isinstance(dtype, cudf.core.dtypes.DecimalDtype):
            _set_decimal_from_scalar(
                self.c_value, value, dtype, valid)
        elif isinstance(dtype, cudf.ListDtype):
            _set_list_from_pylist(
                self.c_value, value, dtype, valid)
        elif isinstance(dtype, cudf.StructDtype):
            _set_struct_from_pydict(self.c_value, value, dtype, valid)
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
        is_datetime = cudf.api.types.is_datetime64_dtype(self.dtype)
        is_timedelta = cudf.api.types.is_timedelta64_dtype(self.dtype)

        null_type = NaT if is_datetime or is_timedelta else NA

        ps = to_arrow_scalar(self)
        if not ps.is_valid:
            return null_type

        # TODO: The special handling of specific types below does not currently
        # extend to nested types containing those types (e.g. List[timedelta]
        # where the timedelta would overflow). We should eventually account for
        # those cases, but that will require more careful consideration of how
        # to traverse the contents of the nested data.
        if is_datetime or is_timedelta:
            time_unit, _ = np.datetime_data(self.dtype)
            # Cast to int64 to avoid overflow
            ps_cast = ps.cast('int64').as_py()
            out_type = np.datetime64 if is_datetime else np.timedelta64
            ret = out_type(ps_cast, time_unit)
        elif cudf.api.types.is_numeric_dtype(self.dtype):
            ret = ps.type.to_pandas_dtype()(ps.as_py())
        else:
            ret = ps.as_py()

        _replace_nested_none(ret)
        return ret

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

        s.c_value = move(ptr)
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

cdef _set_decimal_from_scalar(unique_ptr[scalar]& s,
                              object value,
                              object dtype,
                              bool valid=True):
    value = cudf.utils.dtypes._decimal_to_int64(value) if valid else 0
    if isinstance(dtype, cudf.Decimal64Dtype):
        s.reset(
            new fixed_point_scalar[decimal64](
                <int64_t>np.int64(value), scale_type(-dtype.scale), valid
            )
        )
    elif isinstance(dtype, cudf.Decimal32Dtype):
        s.reset(
            new fixed_point_scalar[decimal32](
                <int32_t>np.int32(value), scale_type(-dtype.scale), valid
            )
        )
    elif isinstance(dtype, cudf.Decimal128Dtype):
        s.reset(
            new fixed_point_scalar[decimal128](
                <libcudf_types.int128>value, scale_type(-dtype.scale), valid
            )
        )
    else:
        raise ValueError(f"dtype not supported: {dtype}")

cdef _set_struct_from_pydict(unique_ptr[scalar]& s,
                             object value,
                             object dtype,
                             bool valid=True):
    arrow_schema = dtype.to_arrow()
    columns = [str(i) for i in range(len(arrow_schema))]
    if valid:
        pyarrow_table = pa.Table.from_arrays(
            [
                pa.array([value[f.name]], from_pandas=True, type=f.type)
                for f in arrow_schema
            ],
            names=columns
        )
    else:
        pyarrow_table = pa.Table.from_arrays(
            [
                pa.array([NA], from_pandas=True, type=f.type)
                for f in arrow_schema
            ],
            names=columns
        )

    data = from_arrow(pyarrow_table)
    cdef table_view struct_view = table_view_from_columns(data)

    s.reset(
        new struct_scalar(struct_view, valid)
    )

cdef _set_list_from_pylist(unique_ptr[scalar]& s,
                           object value,
                           object dtype,
                           bool valid=True):

    value = value if valid else [NA]
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
            _set_datetime64_from_np_scalar(result.c_value, nat, dtype, True)
        elif dtype.type == np.timedelta64:
            _set_timedelta64_from_np_scalar(result.c_value, nat, dtype, True)
        return result
    else:
        raise TypeError('NAT only valid for datetime and timedelta')
