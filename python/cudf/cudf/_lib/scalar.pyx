# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import copy

import numpy as np
import pandas as pd
import pyarrow as pa

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

import pylibcudf

import cudf
from cudf._lib.types import LIBCUDF_TO_SUPPORTED_NUMPY_TYPES
from cudf.core.dtypes import ListDtype, StructDtype
from cudf.core.missing import NA, NaT

cimport pylibcudf.libcudf.types as libcudf_types
# We currently need this cimport because some of the implementations here
# access the c_obj of the scalar, and because we need to be able to call
# pylibcudf.Scalar.from_libcudf. Both of those are temporarily acceptable until
# DeviceScalar is phased out entirely from cuDF Cython (at which point
# cudf.Scalar will be directly backed by pylibcudf.Scalar).
from pylibcudf cimport Scalar as plc_Scalar
from pylibcudf.libcudf.scalar.scalar cimport list_scalar, scalar, struct_scalar

from cudf._lib.types cimport dtype_from_column_view, underlying_type_t_type_id


def _replace_nested(obj, check, replacement):
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if check(item):
                obj[i] = replacement
            elif isinstance(item, (dict, list)):
                _replace_nested(item, check, replacement)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if check(v):
                obj[k] = replacement
            elif isinstance(v, (dict, list)):
                _replace_nested(v, check, replacement)


def gather_metadata(dtypes):
    """Convert a dict of dtypes to a list of ColumnMetadata objects.

    The metadata is constructed recursively so that nested types are
    represented as nested ColumnMetadata objects.

    Parameters
    ----------
    dtypes : dict
        A dict mapping column names to dtypes.

    Returns
    -------
    List[ColumnMetadata]
        A list of ColumnMetadata objects.
    """
    out = []
    for name, dtype in dtypes.items():
        v = pylibcudf.interop.ColumnMetadata(name)
        if isinstance(dtype, cudf.StructDtype):
            v.children_meta = gather_metadata(dtype.fields)
        elif isinstance(dtype, cudf.ListDtype):
            # Offsets column is unnamed and has no children
            v.children_meta.append(pylibcudf.interop.ColumnMetadata(""))
            v.children_meta.extend(
                gather_metadata({"": dtype.element_type})
            )
        out.append(v)
    return out


cdef class DeviceScalar:

    # TODO: I think this should be removable, except that currently the way
    # that from_unique_ptr is implemented is probably dereferencing this in an
    # invalid state. See what the best way to fix that is.
    def __cinit__(self, *args, **kwargs):
        self.c_value = pylibcudf.Scalar.__new__(pylibcudf.Scalar)

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
            _replace_nested(value, cudf.utils.utils.is_na_like, None)

        if isinstance(dtype, cudf.core.dtypes._BaseDtype):
            pa_type = dtype.to_arrow()
        elif pd.api.types.is_string_dtype(dtype):
            # Have to manually convert object types, which we use internally
            # for strings but pyarrow only supports as unicode 'U'
            pa_type = pa.string()
        else:
            pa_type = pa.from_numpy_dtype(dtype)

        if isinstance(pa_type, pa.ListType) and value is None:
            # pyarrow doesn't correctly handle None values for list types, so
            # we have to create this one manually.
            # https://github.com/apache/arrow/issues/40319
            pa_array = pa.array([None], type=pa_type)
        else:
            pa_array = pa.array([pa.scalar(value, type=pa_type)])

        pa_table = pa.Table.from_arrays([pa_array], names=[""])
        table = pylibcudf.interop.from_arrow(pa_table)

        column = table.columns()[0]
        if isinstance(dtype, cudf.core.dtypes.DecimalDtype):
            if isinstance(dtype, cudf.core.dtypes.Decimal32Dtype):
                column = pylibcudf.unary.cast(
                    column, pylibcudf.DataType(pylibcudf.TypeId.DECIMAL32, -dtype.scale)
                )
            elif isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
                column = pylibcudf.unary.cast(
                    column, pylibcudf.DataType(pylibcudf.TypeId.DECIMAL64, -dtype.scale)
                )

        self.c_value = pylibcudf.copying.get_element(column, 0)
        self._dtype = dtype

    def _to_host_scalar(self):
        is_datetime = self.dtype.kind == "M"
        is_timedelta = self.dtype.kind == "m"

        null_type = NaT if is_datetime or is_timedelta else NA

        metadata = gather_metadata({"": self.dtype})[0]
        ps = pylibcudf.interop.to_arrow(self.c_value, metadata)
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

        _replace_nested(ret, lambda item: item is None, NA)
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
        return (<plc_Scalar> self.c_value).c_obj.get()

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
        # Note: This line requires pylibcudf to be cimported
        s.c_value = plc_Scalar.from_libcudf(move(ptr))
        s._set_dtype(dtype)
        return s

    @staticmethod
    cdef DeviceScalar from_pylibcudf(pscalar, dtype=None):
        cdef DeviceScalar s = DeviceScalar.__new__(DeviceScalar)
        s.c_value = pscalar
        s._set_dtype(dtype)
        return s

    cdef void _set_dtype(self, dtype=None):
        cdef libcudf_types.data_type cdtype = self.get_raw_ptr()[0].type()

        if dtype is not None:
            self._dtype = dtype
        elif cdtype.id() in {
            libcudf_types.type_id.DECIMAL32,
            libcudf_types.type_id.DECIMAL64,
            libcudf_types.type_id.DECIMAL128,
        }:
            raise TypeError(
                "Must pass a dtype when constructing from a fixed-point scalar"
            )
        elif cdtype.id() == libcudf_types.type_id.STRUCT:
            struct_table_view = (<struct_scalar*>self.get_raw_ptr())[0].view()
            self._dtype = StructDtype({
                str(i): dtype_from_column_view(struct_table_view.column(i))
                for i in range(struct_table_view.num_columns())
            })
        elif cdtype.id() == libcudf_types.type_id.LIST:
            if (
                <list_scalar*>self.get_raw_ptr()
            )[0].view().type().id() == libcudf_types.type_id.LIST:
                self._dtype = dtype_from_column_view(
                    (<list_scalar*>self.get_raw_ptr())[0].view()
                )
            else:
                self._dtype = ListDtype(
                    LIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
                        <underlying_type_t_type_id>(
                            (<list_scalar*>self.get_raw_ptr())[0]
                            .view().type().id()
                        )
                    ]
                )
        else:
            self._dtype = LIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
                <underlying_type_t_type_id>(cdtype.id())
            ]


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
    elif (isinstance(slr, (np.datetime64, np.timedelta64)) and np.isnat(slr)) or \
            slr is pd.NaT:
        return True
    else:
        return False
