# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import annotations

import builtins
import pickle
import warnings
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda

import cudf
from cudf import _lib as libcudf
from cudf._lib.column import Column
from cudf._lib.null_mask import (
    MaskState,
    bitmask_allocation_size_bytes,
    create_null_mask,
)
from cudf._lib.scalar import as_device_scalar
from cudf._lib.stream_compaction import distinct_count as cpp_distinct_count
from cudf._lib.transform import bools_to_mask
from cudf._typing import BinaryOperand, ColumnLike, Dtype, ScalarLike
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    _is_scalar_or_zero_d_array,
    infer_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_decimal32_dtype,
    is_decimal64_dtype,
    is_decimal_dtype,
    is_dtype_equal,
    is_integer_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_scalar,
    is_string_dtype,
    is_struct_dtype,
    pandas_dtype,
)
from cudf.core.abc import Serializable
from cudf.core.buffer import Buffer
from cudf.core.dtypes import (
    CategoricalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.utils import ioutils, utils
from cudf.utils.dtypes import (
    check_cast_unsupported_dtype,
    cudf_dtype_from_pa_type,
    get_time_unit,
    min_unsigned_type,
    np_to_pa_dtype,
)
from cudf.utils.utils import mask_dtype

T = TypeVar("T", bound="ColumnBase")


class ColumnBase(Column, Serializable):
    def as_frame(self) -> "cudf.core.frame.Frame":
        """
        Converts a Column to Frame
        """
        return cudf.core.frame.Frame({None: self.copy(deep=False)})

    @property
    def data_array_view(self) -> "cuda.devicearray.DeviceNDArray":
        """
        View the data as a device array object
        """
        return cuda.as_cuda_array(self.data).view(self.dtype)

    @property
    def mask_array_view(self) -> "cuda.devicearray.DeviceNDArray":
        """
        View the mask as a device array
        """
        return cuda.as_cuda_array(self.mask).view(mask_dtype)

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        return (
            f"{object.__repr__(self)}\n"
            f"{self.to_arrow().to_string()}\n"
            f"dtype: {self.dtype}"
        )

    def to_pandas(self, index: pd.Index = None, **kwargs) -> "pd.Series":
        """Convert object to pandas type.

        The default implementation falls back to PyArrow for the conversion.
        """
        # This default implementation does not handle nulls in any meaningful
        # way, but must consume the parameter to avoid passing it to PyArrow
        # (which does not recognize it).
        kwargs.pop("nullable", None)
        pd_series = self.to_arrow().to_pandas(**kwargs)

        if index is not None:
            pd_series.index = index
        return pd_series

    def __iter__(self):
        cudf.utils.utils.raise_iteration_error(obj=self)

    @property
    def values_host(self) -> "np.ndarray":
        """
        Return a numpy representation of the Column.
        """
        return self.data_array_view.copy_to_host()

    @property
    def values(self) -> "cupy.ndarray":
        """
        Return a CuPy representation of the Column.
        """
        if len(self) == 0:
            return cupy.asarray([], dtype=self.dtype)

        if self.has_nulls:
            raise ValueError("Column must have no nulls.")

        return cupy.asarray(self.data_array_view)

    def find_and_replace(
        self: T,
        to_replace: ColumnLike,
        replacement: ColumnLike,
        all_nan: bool = False,
    ) -> T:
        raise NotImplementedError

    def clip(self, lo: ScalarLike, hi: ScalarLike) -> ColumnBase:
        return libcudf.replace.clip(self, lo, hi)

    def equals(self, other: ColumnBase, check_dtypes: bool = False) -> bool:
        if self is other:
            return True
        if other is None or len(self) != len(other):
            return False
        if check_dtypes:
            if self.dtype != other.dtype:
                return False
        null_equals = self._null_equals(other)
        return null_equals.all()

    def _null_equals(self, other: ColumnBase) -> ColumnBase:
        return self.binary_operator("NULL_EQUALS", other)

    def all(self) -> bool:
        return bool(libcudf.reduce.reduce("all", self, dtype=np.bool_))

    def any(self) -> bool:
        return bool(libcudf.reduce.reduce("any", self, dtype=np.bool_))

    def __sizeof__(self) -> int:
        n = 0
        if self.data is not None:
            n += self.data.size
        if self.nullable:
            n += bitmask_allocation_size_bytes(self.size)
        return n

    def dropna(self, drop_nan: bool = False) -> ColumnBase:
        if drop_nan:
            col = self.nans_to_nulls()
        else:
            col = self
        dropped_col = (
            col.as_frame()._drop_na_rows(drop_nan=drop_nan)._as_column()
        )
        return dropped_col

    def to_arrow(self) -> pa.Array:
        """Convert to PyArrow Array

        Examples
        --------
        >>> import cudf
        >>> col = cudf.core.column.as_column([1, 2, 3, 4])
        >>> col.to_arrow()
        <pyarrow.lib.Int64Array object at 0x7f886547f830>
        [
          1,
          2,
          3,
          4
        ]
        """
        return libcudf.interop.to_arrow(
            libcudf.table.Table(
                cudf.core.column_accessor.ColumnAccessor({"None": self})
            ),
            [["None"]],
            keep_index=False,
        )["None"].chunk(0)

    @classmethod
    def from_arrow(cls, array: pa.Array) -> ColumnBase:
        """
        Convert PyArrow Array/ChunkedArray to column

        Parameters
        ----------
        array : PyArrow Array/ChunkedArray

        Returns
        -------
        column

        Examples
        --------
        >>> import pyarrow as pa
        >>> import cudf
        >>> cudf.core.column.ColumnBase.from_arrow(pa.array([1, 2, 3, 4]))
        <cudf.core.column.numerical.NumericalColumn object at 0x7f8865497ef0>
        """
        if not isinstance(array, (pa.Array, pa.ChunkedArray)):
            raise TypeError("array should be PyArrow array or chunked array")
        data = pa.table([array], [None])
        if isinstance(array.type, pa.DictionaryType):
            indices_table = pa.table(
                {
                    "None": pa.chunked_array(
                        [chunk.indices for chunk in data["None"].chunks],
                        type=array.type.index_type,
                    )
                }
            )
            dictionaries_table = pa.table(
                {
                    "None": pa.chunked_array(
                        [chunk.dictionary for chunk in data["None"].chunks],
                        type=array.type.value_type,
                    )
                }
            )

            codes = libcudf.interop.from_arrow(
                indices_table, indices_table.column_names
            )._data["None"]
            categories = libcudf.interop.from_arrow(
                dictionaries_table, dictionaries_table.column_names
            )._data["None"]

            return build_categorical_column(
                categories=categories,
                codes=codes,
                mask=codes.base_mask,
                size=codes.size,
                ordered=array.type.ordered,
            )
        elif isinstance(array.type, pa.StructType):
            return cudf.core.column.StructColumn.from_arrow(array)
        elif isinstance(
            array.type, pd.core.arrays._arrow_utils.ArrowIntervalType
        ):
            return cudf.core.column.IntervalColumn.from_arrow(array)
        elif isinstance(array.type, pa.Decimal128Type):
            return cudf.core.column.Decimal64Column.from_arrow(array)

        result = libcudf.interop.from_arrow(data, data.column_names)._data[
            "None"
        ]

        result = result._with_type_metadata(
            cudf_dtype_from_pa_type(array.type)
        )
        return result

    def _get_mask_as_column(self) -> ColumnBase:
        return libcudf.transform.mask_to_bools(
            self.base_mask, self.offset, self.offset + len(self)
        )

    def _memory_usage(self, **kwargs) -> int:
        return self.__sizeof__()

    def default_na_value(self) -> Any:
        raise NotImplementedError()

    def to_gpu_array(self, fillna=None) -> "cuda.devicearray.DeviceNDArray":
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : scalar, 'pandas', or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        if fillna:
            return self.fillna(self.default_na_value()).data_array_view
        else:
            return self.dropna(drop_nan=False).data_array_view

    def to_array(self, fillna=None) -> np.ndarray:
        """Get a dense numpy array for the data.

        Parameters
        ----------
        fillna : scalar, 'pandas', or None
            Defaults to None, which will skip null values.
            If it equals "pandas", null values are filled with NaNs.
            Non integral dtype is promoted to np.float64.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """

        return self.to_gpu_array(fillna=fillna).copy_to_host()

    def _reverse(self):
        return libcudf.copying.reverse(self)

    def _fill(
        self,
        fill_value: ScalarLike,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Optional[ColumnBase]:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        fill_scalar = as_device_scalar(fill_value, self.dtype)

        if not inplace:
            return libcudf.filling.fill(self, begin, end, fill_scalar)

        if is_string_dtype(self.dtype):
            return self._mimic_inplace(
                libcudf.filling.fill(self, begin, end, fill_scalar),
                inplace=True,
            )

        if fill_value is None and not self.nullable:
            mask = create_null_mask(self.size, state=MaskState.ALL_VALID)
            self.set_base_mask(mask)

        libcudf.filling.fill_in_place(self, begin, end, fill_scalar)

        return self

        fill_code = self._encode(fill_value)
        fill_scalar = as_device_scalar(fill_code, self.codes.dtype)

        result = self if inplace else self.copy()

        libcudf.filling.fill_in_place(result.codes, begin, end, fill_scalar)
        return result

    def shift(self, offset: int, fill_value: ScalarLike) -> ColumnBase:
        return libcudf.copying.shift(self, offset, fill_value)

    @property
    def valid_count(self) -> int:
        """Number of non-null values"""
        return len(self) - self.null_count

    @property
    def nullmask(self) -> Buffer:
        """The gpu buffer for the null-mask
        """
        if self.nullable:
            return self.mask_array_view
        else:
            raise ValueError("Column has no null mask")

    def copy(self: T, deep: bool = True) -> T:
        """Columns are immutable, so a deep copy produces a copy of the
        underlying data and mask and a shallow copy creates a new column and
        copies the references of the data and mask.
        """
        if deep:
            result = libcudf.copying.copy_column(self)
            return cast(T, result._with_type_metadata(self.dtype))
        else:
            return cast(
                T,
                build_column(
                    self.base_data,
                    self.dtype,
                    mask=self.base_mask,
                    size=self.size,
                    offset=self.offset,
                    children=self.base_children,
                ),
            )

    def view(self, dtype: Dtype) -> ColumnBase:
        """
        View the data underlying a column as different dtype.
        The source column must divide evenly into the size of
        the desired data type. Columns with nulls may only be
        viewed as dtypes with size equal to source dtype size

        Parameters
        ----------
        dtype : NumPy dtype, string
            The dtype to view the data as

        """

        dtype = np.dtype(dtype)

        if dtype.kind in ("o", "u", "s"):
            raise TypeError(
                "Bytes viewed as str without metadata is ambiguous"
            )

        if self.dtype.itemsize == dtype.itemsize:
            return build_column(
                self.base_data,
                dtype=dtype,
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
            )

        else:
            if self.null_count > 0:
                raise ValueError(
                    "Can not produce a view of a column with nulls"
                )

            if (self.size * self.dtype.itemsize) % dtype.itemsize:
                raise ValueError(
                    f"Can not divide {self.size * self.dtype.itemsize}"
                    + f" total bytes into {dtype} with size {dtype.itemsize}"
                )

            assert self.base_data is not None
            new_buf_ptr = (
                self.base_data.ptr + self.offset * self.dtype.itemsize
            )
            new_buf_size = self.size * self.dtype.itemsize
            view_buf = Buffer(
                data=new_buf_ptr,
                size=new_buf_size,
                owner=self.base_data._owner,
            )
            return build_column(view_buf, dtype=dtype)

    def element_indexing(self, index: int):
        """Default implementation for indexing to an element

        Raises
        ------
        ``IndexError`` if out-of-bound
        """
        idx = np.int32(index)
        if idx < 0:
            idx = len(self) + idx
        if idx > len(self) - 1 or idx < 0:
            raise IndexError("single positional indexer is out-of-bounds")

        return libcudf.copying.get_element(self, idx).value

    def slice(self, start: int, stop: int, stride: int = None) -> ColumnBase:
        stride = 1 if stride is None else stride
        if start < 0:
            start = start + len(self)
        if stop < 0 and not (stride < 0 and stop == -1):
            stop = stop + len(self)
        if (stride > 0 and start >= stop) or (stride < 0 and start <= stop):
            return column_empty(0, self.dtype, masked=True)
        # compute mask slice
        if stride == 1:
            return libcudf.copying.column_slice(self, [start, stop])[0]
        else:
            # Need to create a gather map for given slice with stride
            gather_map = arange(
                start=start, stop=stop, step=stride, dtype=np.dtype(np.int32),
            )
            return self.take(gather_map)

    def __getitem__(self, arg) -> Union[ScalarLike, ColumnBase]:
        if _is_scalar_or_zero_d_array(arg):
            return self.element_indexing(int(arg))
        elif isinstance(arg, slice):
            start, stop, stride = arg.indices(len(self))
            return self.slice(start, stop, stride)
        else:
            arg = as_column(arg)
            if len(arg) == 0:
                arg = as_column([], dtype="int32")
            if is_integer_dtype(arg.dtype):
                return self.take(arg)
            if is_bool_dtype(arg.dtype):
                return self.apply_boolean_mask(arg)
            raise NotImplementedError(type(arg))

    def __setitem__(self, key: Any, value: Any):
        """
        Set the value of self[key] to value.

        If value and self are of different types,
        value is coerced to self.dtype
        """

        if isinstance(key, slice):
            key_start, key_stop, key_stride = key.indices(len(self))
            if key_start < 0:
                key_start = key_start + len(self)
            if key_stop < 0:
                key_stop = key_stop + len(self)
            if key_start >= key_stop:
                return self.copy()
            if (key_stride is None or key_stride == 1) and is_scalar(value):
                return self._fill(value, key_start, key_stop, inplace=True)
            if key_stride != 1 or key_stride is not None or is_scalar(value):
                key = arange(
                    start=key_start,
                    stop=key_stop,
                    step=key_stride,
                    dtype=np.dtype(np.int32),
                )
                nelem = len(key)
            else:
                nelem = abs(key_stop - key_start)
        else:
            key = as_column(key)
            if is_bool_dtype(key.dtype):
                if not len(key) == len(self):
                    raise ValueError(
                        "Boolean mask must be of same length as column"
                    )
                key = arange(len(self))[key]
                if hasattr(value, "__len__") and len(value) == len(self):
                    value = as_column(value)[key]
            nelem = len(key)

        if is_scalar(value):
            value = cudf.Scalar(value, dtype=self.dtype)
        else:
            if len(value) != nelem:
                msg = (
                    f"Size mismatch: cannot set value "
                    f"of size {len(value)} to indexing result of size "
                    f"{nelem}"
                )
                raise ValueError(msg)
            value = as_column(value).astype(self.dtype)

        if (
            isinstance(key, slice)
            and (key_stride == 1 or key_stride is None)
            and not is_scalar(value)
        ):

            out = libcudf.copying.copy_range(
                value, self, 0, nelem, key_start, key_stop, False
            )
        else:
            try:
                if is_scalar(value):
                    input = self
                    out = input.as_frame()._scatter(key, [value])._as_column()
                else:
                    if not isinstance(value, Column):
                        value = as_column(value)
                    out = (
                        self.as_frame()
                        ._scatter(key, value.as_frame())
                        ._as_column()
                    )
            except RuntimeError as e:
                if "out of bounds" in str(e):
                    raise IndexError(
                        f"index out of bounds for column of size {len(self)}"
                    ) from e
                raise

        self._mimic_inplace(out, inplace=True)

    def fillna(
        self: T,
        value: Any = None,
        method: builtins.str = None,
        dtype: Dtype = None,
    ) -> T:
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        return libcudf.replace.replace_nulls(
            input_col=self, replacement=value, method=method, dtype=dtype
        )

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column.
        """
        result = libcudf.unary.is_null(self)

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values incase
            # of a float column
            result = result | libcudf.unary.is_nan(self)

        return result

    def isna(self) -> ColumnBase:
        """Identify missing values in a Column. Alias for isnull.
        """
        return self.isnull()

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column.
        """
        result = libcudf.unary.is_valid(self)

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values incase
            # of a float column
            result = result & libcudf.unary.is_non_nan(self)

        return result

    def notna(self) -> ColumnBase:
        """Identify non-missing values in a Column. Alias for notnull.
        """
        return self.notnull()

    def find_first_value(
        self, value: ScalarLike, closest: bool = False
    ) -> int:
        """
        Returns offset of first value that matches
        """
        # FIXME: Inefficient, may be need a libcudf api
        index = cudf.core.index.RangeIndex(0, stop=len(self))
        indices = index.take(self == value)
        if not len(indices):
            raise ValueError("value not found")
        return indices[0]

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        """
        Returns offset of last value that matches
        """
        # FIXME: Inefficient, may be need a libcudf api
        index = cudf.core.index.RangeIndex(0, stop=len(self))
        indices = index.take(self == value)
        if not len(indices):
            raise ValueError("value not found")
        return indices[-1]

    def append(self, other: ColumnBase) -> ColumnBase:
        return _concat_columns([self, as_column(other)])

    def quantile(
        self,
        q: Union[float, Sequence[float]],
        interpolation: builtins.str,
        exact: bool,
    ) -> ColumnBase:
        raise TypeError(f"cannot perform quantile with type {self.dtype}")

    def median(self, skipna: bool = None) -> ScalarLike:
        raise TypeError(f"cannot perform median with type {self.dtype}")

    def take(
        self: T,
        indices: ColumnBase,
        keep_index: bool = True,
        nullify: bool = False,
    ) -> T:
        """Return Column by taking values from the corresponding *indices*.
        """
        # Handle zero size
        if indices.size == 0:
            return cast(T, column_empty_like(self, newsize=0))
        try:
            return (
                self.as_frame()
                ._gather(indices, keep_index=keep_index, nullify=nullify)
                ._as_column()
            )
        except RuntimeError as e:
            if "out of bounds" in str(e):
                raise IndexError(
                    f"index out of bounds for column of size {len(self)}"
                ) from e
            raise

    def isin(self, values: Sequence) -> ColumnBase:
        """Check whether values are contained in the Column.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a TypeError. Instead, turn a single string into a list
            of one element.

        Returns
        -------
        result: Column
            Column of booleans indicating if each element is in values.
        """
        try:
            lhs, rhs = self._process_values_for_isin(values)
            res = lhs._isin_earlystop(rhs)
            if res is not None:
                return res
        except ValueError:
            # pandas functionally returns all False when cleansing via
            # typecasting fails
            return full(len(self), False, dtype="bool")

        res = lhs._obtain_isin_result(rhs)

        return res

    def _process_values_for_isin(
        self, values: Sequence
    ) -> Tuple[ColumnBase, ColumnBase]:
        """
        Helper function for `isin` which pre-process `values` based on `self`.
        """
        lhs = self
        rhs = as_column(values, nan_as_null=False)
        if lhs.null_count == len(lhs):
            lhs = lhs.astype(rhs.dtype)
        elif rhs.null_count == len(rhs):
            rhs = rhs.astype(lhs.dtype)
        return lhs, rhs

    def _isin_earlystop(self, rhs: ColumnBase) -> Union[ColumnBase, None]:
        """
        Helper function for `isin` which determines possibility of
        early-stopping or not.
        """
        if self.dtype != rhs.dtype:
            if self.null_count and rhs.null_count:
                return self.isna()
            else:
                return cudf.core.column.full(len(self), False, dtype="bool")
        elif self.null_count == 0 and (rhs.null_count == len(rhs)):
            return cudf.core.column.full(len(self), False, dtype="bool")
        else:
            return None

    def _obtain_isin_result(self, rhs: ColumnBase) -> ColumnBase:
        """
        Helper function for `isin` which merges `self` & `rhs`
        to determine what values of `rhs` exist in `self`.
        """
        ldf = cudf.DataFrame({"x": self, "orig_order": arange(len(self))})
        rdf = cudf.DataFrame(
            {"x": rhs, "bool": full(len(rhs), True, dtype="bool")}
        )
        res = ldf.merge(rdf, on="x", how="left").sort_values(by="orig_order")
        res = res.drop_duplicates(subset="orig_order", ignore_index=True)
        res = res._data["bool"].fillna(False)
        return res

    def as_mask(self) -> Buffer:
        """Convert booleans to bitmask

        Returns
        -------
        Buffer
        """

        if self.has_nulls:
            raise ValueError("Column must have no nulls.")

        return bools_to_mask(self)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""

        return cudf.io.dlpack.to_dlpack(self)

    @property
    def is_unique(self) -> bool:
        return self.distinct_count() == len(self)

    @property
    def is_monotonic(self) -> bool:
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self) -> bool:
        return not self.has_nulls and self.as_frame()._is_sorted(
            ascending=None, null_position=None
        )

    @property
    def is_monotonic_decreasing(self) -> bool:
        return not self.has_nulls and self.as_frame()._is_sorted(
            ascending=[False], null_position=None
        )

    def get_slice_bound(
        self, label: ScalarLike, side: builtins.str, kind: builtins.str
    ) -> int:
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.
        Parameters
        ----------
        label : Scalar
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}
        """
        assert kind in ["ix", "loc", "getitem", None]
        if side not in ("left", "right"):
            raise ValueError(
                "Invalid value for side kwarg,"
                " must be either 'left' or 'right': %s" % (side,)
            )

        # TODO: Handle errors/missing keys correctly
        #       Not currently using `kind` argument.
        if side == "left":
            return self.find_first_value(label, closest=True)
        elif side == "right":
            return self.find_last_value(label, closest=True) + 1
        else:
            raise ValueError(f"Invalid value for side: {side}")

    def sort_by_values(
        self: ColumnBase,
        ascending: bool = True,
        na_position: builtins.str = "last",
    ) -> Tuple[ColumnBase, "cudf.core.column.NumericalColumn"]:
        col_inds = self.as_frame()._get_sorted_inds(
            ascending=ascending, na_position=na_position
        )
        col_keys = self.take(col_inds)
        return col_keys, col_inds

    def distinct_count(
        self, method: builtins.str = "sort", dropna: bool = True
    ) -> int:
        if method != "sort":
            msg = "non sort based distinct_count() not implemented yet"
            raise NotImplementedError(msg)
        return cpp_distinct_count(self, ignore_nulls=dropna)

    def can_cast_safely(self, to_dtype: Dtype) -> bool:
        raise NotImplementedError()

    def astype(self, dtype: Dtype, **kwargs) -> ColumnBase:
        if _is_non_decimal_numeric_dtype(dtype):
            return self.as_numerical_column(dtype, **kwargs)
        elif is_categorical_dtype(dtype):
            return self.as_categorical_column(dtype, **kwargs)
        elif pandas_dtype(dtype).type in {
            np.str_,
            np.object_,
            str,
        }:
            return self.as_string_column(dtype, **kwargs)
        elif is_list_dtype(dtype):
            if not self.dtype == dtype:
                raise NotImplementedError(
                    "Casting list columns not currently supported"
                )
            return self
        elif is_struct_dtype(dtype):
            if not self.dtype == dtype:
                raise NotImplementedError(
                    "Casting struct columns not currently supported"
                )
            return self
        elif is_interval_dtype(self.dtype):
            return self.as_interval_column(dtype, **kwargs)
        elif is_decimal_dtype(dtype):
            return self.as_decimal_column(dtype, **kwargs)
        elif np.issubdtype(dtype, np.datetime64):
            return self.as_datetime_column(dtype, **kwargs)
        elif np.issubdtype(dtype, np.timedelta64):
            return self.as_timedelta_column(dtype, **kwargs)
        else:
            return self.as_numerical_column(dtype, **kwargs)

    def as_categorical_column(self, dtype, **kwargs) -> ColumnBase:
        if "ordered" in kwargs:
            ordered = kwargs["ordered"]
        else:
            ordered = False

        sr = cudf.Series(self)

        # Re-label self w.r.t. the provided categories
        if isinstance(dtype, (cudf.CategoricalDtype, pd.CategoricalDtype)):
            labels = sr.label_encoding(cats=dtype.categories)
            if "ordered" in kwargs:
                warnings.warn(
                    "Ignoring the `ordered` parameter passed in `**kwargs`, "
                    "will be using `ordered` parameter of CategoricalDtype"
                )

            return build_categorical_column(
                categories=dtype.categories,
                codes=labels._column,
                mask=self.mask,
                ordered=dtype.ordered,
            )

        cats = sr.unique().astype(sr.dtype)
        label_dtype = min_unsigned_type(len(cats))
        labels = sr.label_encoding(cats=cats, dtype=label_dtype, na_sentinel=1)

        # columns include null index in factorization; remove:
        if self.has_nulls:
            cats = cats._column.dropna(drop_nan=False)
            min_type = min_unsigned_type(len(cats), 8)
            labels = labels - 1
            if np.dtype(min_type).itemsize < labels.dtype.itemsize:
                labels = labels.astype(min_type)

        return build_categorical_column(
            categories=cats,
            codes=labels._column,
            mask=self.mask,
            ordered=ordered,
        )

    def as_numerical_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.NumericalColumn":
        raise NotImplementedError

    def as_datetime_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.DatetimeColumn":
        raise NotImplementedError

    def as_interval_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.IntervalColumn":
        raise NotImplementedError

    def as_timedelta_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.TimeDeltaColumn":
        raise NotImplementedError

    def as_string_column(
        self, dtype: Dtype, format=None, **kwargs
    ) -> "cudf.core.column.StringColumn":
        raise NotImplementedError

    def as_decimal_column(
        self, dtype: Dtype, **kwargs
    ) -> Union[
        "cudf.core.column.Decimal32Column", "cudf.core.column.Decimal64Column"
    ]:
        raise NotImplementedError

    def as_decimal64_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.Decimal64Column":
        raise NotImplementedError

    def as_decimal32_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.Decimal32Column":
        raise NotImplementedError

    def apply_boolean_mask(self, mask) -> ColumnBase:
        mask = as_column(mask, dtype="bool")
        result = (
            self.as_frame()._apply_boolean_mask(boolean_mask=mask)._as_column()
        )
        return result

    def argsort(
        self, ascending: bool = True, na_position: builtins.str = "last"
    ) -> ColumnBase:

        sorted_indices = self.as_frame()._get_sorted_inds(
            ascending=ascending, na_position=na_position
        )
        return sorted_indices

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed. To explicitly construct a host array, consider using "
            ".to_array()"
        )

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            f"dtype {self.dtype} is not yet supported via "
            "`__cuda_array_interface__`"
        )

    def __add__(self, other):
        return self.binary_operator("add", other)

    def __sub__(self, other):
        return self.binary_operator("sub", other)

    def __mul__(self, other):
        return self.binary_operator("mul", other)

    def __eq__(self, other):
        return self.binary_operator("eq", other)

    def __ne__(self, other):
        return self.binary_operator("ne", other)

    def __or__(self, other):
        return self.binary_operator("or", other)

    def __and__(self, other):
        return self.binary_operator("and", other)

    def __floordiv__(self, other):
        return self.binary_operator("floordiv", other)

    def __truediv__(self, other):
        return self.binary_operator("truediv", other)

    def __mod__(self, other):
        return self.binary_operator("mod", other)

    def __pow__(self, other):
        return self.binary_operator("pow", other)

    def __lt__(self, other):
        return self.binary_operator("lt", other)

    def __gt__(self, other):
        return self.binary_operator("gt", other)

    def __le__(self, other):
        return self.binary_operator("le", other)

    def __ge__(self, other):
        return self.binary_operator("ge", other)

    def searchsorted(
        self,
        value,
        side: builtins.str = "left",
        ascending: bool = True,
        na_position: builtins.str = "last",
    ):
        values = as_column(value).as_frame()
        return self.as_frame().searchsorted(
            values, side, ascending=ascending, na_position=na_position
        )

    def unique(self) -> ColumnBase:
        """
        Get unique values in the data
        """
        return (
            self.as_frame()
            .drop_duplicates(keep="first", ignore_index=True)
            ._as_column()
        )

    def serialize(self) -> Tuple[dict, list]:
        header: Dict[Any, Any] = {}
        frames = []
        header["type-serialized"] = pickle.dumps(type(self))
        header["dtype"] = self.dtype.str

        if self.data is not None:
            data_header, data_frames = self.data.serialize()
            header["data"] = data_header
            frames.extend(data_frames)

        if self.mask is not None:
            mask_header, mask_frames = self.mask.serialize()
            header["mask"] = mask_header
            frames.extend(mask_frames)

        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> ColumnBase:
        dtype = header["dtype"]
        data = Buffer.deserialize(header["data"], [frames[0]])
        mask = None
        if "mask" in header:
            mask = Buffer.deserialize(header["mask"], [frames[1]])
        return build_column(
            data=data, dtype=dtype, mask=mask, size=header.get("size", None)
        )

    def unary_operator(self, unaryop: builtins.str):
        raise TypeError(
            f"Operation {unaryop} not supported for dtype {self.dtype}."
        )

    def binary_operator(
        self, op: builtins.str, other: BinaryOperand, reflect: bool = False
    ) -> ColumnBase:
        raise TypeError(
            f"Operation {op} not supported between dtypes {self.dtype} and "
            f"{other.dtype}."
        )

    def normalize_binop_value(
        self, other: ScalarLike
    ) -> Union[ColumnBase, ScalarLike]:
        raise NotImplementedError

    def min(self, skipna: bool = None, dtype: Dtype = None):
        result_col = self._process_for_reduction(skipna=skipna)
        if isinstance(result_col, ColumnBase):
            return libcudf.reduce.reduce("min", result_col, dtype=dtype)
        else:
            return result_col

    def max(self, skipna: bool = None, dtype: Dtype = None):
        result_col = self._process_for_reduction(skipna=skipna)
        if isinstance(result_col, ColumnBase):
            return libcudf.reduce.reduce("max", result_col, dtype=dtype)
        else:
            return result_col

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ):
        raise TypeError(f"cannot perform sum with type {self.dtype}")

    def product(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ):
        raise TypeError(f"cannot perform product with type {self.dtype}")

    def mean(self, skipna: bool = None, dtype: Dtype = None):
        raise TypeError(f"cannot perform mean with type {self.dtype}")

    def std(self, skipna: bool = None, ddof=1, dtype: Dtype = np.float64):
        raise TypeError(f"cannot perform std with type {self.dtype}")

    def var(self, skipna: bool = None, ddof=1, dtype: Dtype = np.float64):
        raise TypeError(f"cannot perform var with type {self.dtype}")

    def kurtosis(self, skipna: bool = None):
        raise TypeError(f"cannot perform kurtosis with type {self.dtype}")

    def skew(self, skipna: bool = None):
        raise TypeError(f"cannot perform skew with type {self.dtype}")

    def cov(self, other: ColumnBase):
        raise TypeError(
            f"cannot perform covarience with types {self.dtype}, "
            f"{other.dtype}"
        )

    def corr(self, other: ColumnBase):
        raise TypeError(
            f"cannot perform corr with types {self.dtype}, {other.dtype}"
        )

    def nans_to_nulls(self: T) -> T:
        if self.dtype.kind == "f":
            newmask = libcudf.transform.nans_to_nulls(self)
            return self.set_mask(newmask)
        else:
            return self

    def _process_for_reduction(
        self, skipna: bool = None, min_count: int = 0
    ) -> Union[ColumnBase, ScalarLike]:
        skipna = True if skipna is None else skipna

        if skipna:
            result_col = self.nans_to_nulls()
            if result_col.has_nulls:
                result_col = result_col.dropna()
        else:
            if self.has_nulls:
                return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

            result_col = self

        if min_count > 0:
            valid_count = len(result_col) - result_col.null_count
            if valid_count < min_count:
                return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
        elif min_count < 0:
            warnings.warn(
                f"min_count value cannot be negative({min_count}), will "
                f"default to 0."
            )
        return result_col

    def scatter_to_table(
        self,
        row_indices: ColumnBase,
        column_indices: ColumnBase,
        names: List[Any],
        nrows: int = None,
        ncols: int = None,
    ) -> "cudf.core.frame.Frame":
        """
        Scatters values from the column into a table.

        Parameters
        ----------
        row_indices
            A column of the same size as `self` specifying the
            row index to scatter each value to
        column_indices
            A column of the same size as `self` specifying the
            column index to scatter each value to
        names
            The column names of the resulting table

        Returns
        -------
        """
        if nrows is None:
            nrows = 0
            if len(row_indices) > 0:
                nrows = int(row_indices.max() + 1)

        if ncols is None:
            ncols = 0
            if len(column_indices) > 0:
                ncols = int(column_indices.max() + 1)

        if nrows * ncols == 0:
            return cudf.core.frame.Frame({})

        scatter_map = (column_indices * np.int32(nrows)) + row_indices
        target = cudf.core.frame.Frame(
            {None: column_empty_like(self, masked=True, newsize=nrows * ncols)}
        )
        target._data[None][scatter_map] = self
        result_frames = target._split(range(nrows, nrows * ncols, nrows))
        return cudf.core.frame.Frame(
            {
                name: next(iter(f._columns))
                for name, f in zip(names, result_frames)
            }
        )

    def _with_type_metadata(self: ColumnBase, dtype: Dtype) -> ColumnBase:
        """
        Copies type metadata from self onto other, returning a new column.

        When ``self`` is a nested column, recursively apply this function on
        the children of ``self``.
        """
        return self


def column_empty_like(
    column: ColumnBase,
    dtype: Dtype = None,
    masked: bool = False,
    newsize: int = None,
) -> ColumnBase:
    """Allocate a new column like the given *column*
    """
    if dtype is None:
        dtype = column.dtype
    row_count = len(column) if newsize is None else newsize

    if (
        hasattr(column, "dtype")
        and is_categorical_dtype(column.dtype)
        and dtype == column.dtype
    ):
        column = cast("cudf.core.column.CategoricalColumn", column)
        codes = column_empty_like(column.codes, masked=masked, newsize=newsize)
        return build_column(
            data=None,
            dtype=dtype,
            mask=codes.base_mask,
            children=(as_column(codes.base_data, dtype=codes.dtype),),
            size=codes.size,
        )

    return column_empty(row_count, dtype, masked)


def column_empty_like_same_mask(
    column: ColumnBase, dtype: Dtype
) -> ColumnBase:
    """Create a new empty Column with the same length and the same mask.

    Parameters
    ----------
    dtype : np.dtype like
        The dtype of the data buffer.
    """
    result = column_empty_like(column, dtype)
    if column.nullable:
        result = result.set_mask(column.mask)
    return result


def column_empty(
    row_count: int, dtype: Dtype = "object", masked: bool = False
) -> ColumnBase:
    """Allocate a new column like the given row_count and dtype.
    """
    dtype = pandas_dtype(dtype)
    children = ()  # type: Tuple[ColumnBase, ...]

    if is_categorical_dtype(dtype):
        data = None
        children = (
            build_column(
                data=Buffer.empty(row_count * np.dtype("int32").itemsize),
                dtype="int32",
            ),
        )
    elif dtype.kind in "OU" and not is_decimal_dtype(dtype):
        data = None
        children = (
            full(row_count + 1, 0, dtype="int32"),
            build_column(
                data=Buffer.empty(row_count * np.dtype("int8").itemsize),
                dtype="int8",
            ),
        )
    else:
        data = Buffer.empty(row_count * dtype.itemsize)

    if masked:
        mask = create_null_mask(row_count, state=MaskState.ALL_NULL)
    else:
        mask = None

    return build_column(
        data, dtype, mask=mask, size=row_count, children=children
    )


def build_column(
    data: Union[Buffer, None],
    dtype: Dtype,
    *,
    size: int = None,
    mask: Buffer = None,
    offset: int = 0,
    null_count: int = None,
    children: Tuple[ColumnBase, ...] = (),
) -> ColumnBase:
    """
    Build a Column of the appropriate type from the given parameters

    Parameters
    ----------
    data : Buffer
        The data buffer (can be None if constructing certain Column
        types like StringColumn, ListColumn, or CategoricalColumn)
    dtype
        The dtype associated with the Column to construct
    mask : Buffer, optional
        The mask buffer
    size : int, optional
    offset : int, optional
    children : tuple, optional
    """
    dtype = pandas_dtype(dtype)

    if _is_non_decimal_numeric_dtype(dtype):
        assert data is not None
        return cudf.core.column.NumericalColumn(
            data=data,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
        )
    if is_categorical_dtype(dtype):
        if not len(children) == 1:
            raise ValueError(
                "Must specify exactly one child column for CategoricalColumn"
            )
        if not isinstance(children[0], ColumnBase):
            raise TypeError("children must be a tuple of Columns")
        return cudf.core.column.CategoricalColumn(
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
            children=children,
        )
    elif dtype.type is np.datetime64:
        if data is None:
            raise TypeError("Must specify data buffer")
        return cudf.core.column.DatetimeColumn(
            data=data,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
        )
    elif dtype.type is np.timedelta64:
        if data is None:
            raise TypeError("Must specify data buffer")
        return cudf.core.column.TimeDeltaColumn(
            data=data,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
        )
    elif dtype.type in (np.object_, np.str_):
        return cudf.core.column.StringColumn(
            mask=mask,
            size=size,
            offset=offset,
            children=children,
            null_count=null_count,
        )
    elif is_list_dtype(dtype):
        return cudf.core.column.ListColumn(
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )
    elif is_interval_dtype(dtype):
        return cudf.core.column.IntervalColumn(
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            children=children,
            null_count=null_count,
        )
    elif is_struct_dtype(dtype):
        if size is None:
            raise TypeError("Must specify size")
        return cudf.core.column.StructColumn(
            data=data,
            dtype=dtype,
            size=size,
            mask=mask,
            null_count=null_count,
            children=children,
        )
    elif is_decimal64_dtype(dtype):
        if size is None:
            raise TypeError("Must specify size")
        return cudf.core.column.Decimal64Column(
            data=data,
            size=size,
            offset=offset,
            dtype=dtype,
            mask=mask,
            null_count=null_count,
            children=children,
        )
    elif is_decimal32_dtype(dtype):
        if size is None:
            raise TypeError("Must specify size")
        return cudf.core.column.Decimal32Column(
            data=data,
            size=size,
            offset=offset,
            dtype=dtype,
            mask=mask,
            null_count=null_count,
            children=children,
        )
    elif is_interval_dtype(dtype):
        return cudf.core.column.IntervalColumn(
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
            children=children,
        )
    else:
        raise TypeError(f"Unrecognized dtype: {dtype}")


def build_categorical_column(
    categories: ColumnBase,
    codes: ColumnBase,
    mask: Buffer = None,
    size: int = None,
    offset: int = 0,
    null_count: int = None,
    ordered: bool = None,
) -> "cudf.core.column.CategoricalColumn":
    """
    Build a CategoricalColumn

    Parameters
    ----------
    categories : Column
        Column of categories
    codes : Column
        Column of codes, the size of the resulting Column will be
        the size of `codes`
    mask : Buffer
        Null mask
    size : int, optional
    offset : int, optional
    ordered : bool
        Indicates whether the categories are ordered
    """
    codes_dtype = min_unsigned_type(len(categories))
    codes = as_column(codes)
    if codes.dtype != codes_dtype:
        codes = codes.astype(codes_dtype)

    dtype = CategoricalDtype(categories=categories, ordered=ordered)

    result = build_column(
        data=None,
        dtype=dtype,
        mask=mask,
        size=size,
        offset=offset,
        null_count=null_count,
        children=(codes,),
    )
    return cast("cudf.core.column.CategoricalColumn", result)


def build_interval_column(
    left_col,
    right_col,
    mask=None,
    size=None,
    offset=0,
    null_count=None,
    closed="right",
):
    """
    Build an IntervalColumn

    Parameters
    ----------
    left_col : Column
        Column of values representing the left of the interval
    right_col : Column
        Column of representing the right of the interval
    mask : Buffer
        Null mask
    size : int, optional
    offset : int, optional
    closed : {"left", "right", "both", "neither"}, default "right"
            Whether the intervals are closed on the left-side, right-side,
            both or neither.
    """
    left = as_column(left_col)
    right = as_column(right_col)
    if closed not in {"left", "right", "both", "neither"}:
        closed = "right"
    if type(left_col) is not list:
        dtype = IntervalDtype(left_col.dtype, closed)
    else:
        dtype = IntervalDtype("int64", closed)
    size = len(left)
    return build_column(
        data=None,
        dtype=dtype,
        mask=mask,
        size=size,
        offset=offset,
        null_count=null_count,
        children=(left, right),
    )


def build_list_column(
    indices: ColumnBase,
    elements: ColumnBase,
    mask: Buffer = None,
    size: int = None,
    offset: int = 0,
    null_count: int = None,
) -> "cudf.core.column.ListColumn":
    """
    Build a ListColumn

    Parameters
    ----------
    indices : ColumnBase
        Column of list indices
    elements : ColumnBase
        Column of list elements
    mask: Buffer
        Null mask
    size: int, optional
    offset: int, optional
    """
    dtype = ListDtype(element_type=elements.dtype)

    result = build_column(
        data=None,
        dtype=dtype,
        mask=mask,
        size=size,
        offset=offset,
        null_count=null_count,
        children=(indices, elements),
    )

    return cast("cudf.core.column.ListColumn", result)


def build_struct_column(
    names: Sequence[str],
    children: Tuple[ColumnBase, ...],
    dtype: Optional[Dtype] = None,
    mask: Buffer = None,
    size: int = None,
    offset: int = 0,
    null_count: int = None,
) -> "cudf.core.column.StructColumn":
    """
    Build a StructColumn

    Parameters
    ----------
    names : list-like
        Field names to map to children dtypes
    children : tuple

    mask: Buffer
        Null mask
    size: int, optional
    offset: int, optional
    """
    if dtype is None:
        dtype = StructDtype(
            fields={name: col.dtype for name, col in zip(names, children)}
        )

    result = build_column(
        data=None,
        dtype=dtype,
        mask=mask,
        size=size,
        offset=offset,
        null_count=null_count,
        children=children,
    )

    return cast("cudf.core.column.StructColumn", result)


def as_column(
    arbitrary: Any,
    nan_as_null: bool = None,
    dtype: Dtype = None,
    length: int = None,
):
    """Create a Column from an arbitrary object

    Parameters
    ----------
    arbitrary : object
        Object to construct the Column from. See *Notes*.
    nan_as_null : bool, optional, default None
        If None (default), treats NaN values in arbitrary as null if there is
        no mask passed along with it. If True, combines the mask and NaNs to
        form a new validity mask. If False, leaves NaN values as is.
    dtype : optional
        Optionally typecast the constructed Column to the given
        dtype.
    length : int, optional
        If `arbitrary` is a scalar, broadcast into a Column of
        the given length.

    Returns
    -------
    A Column of the appropriate type and size.

    Notes
    -----
    Currently support inputs are:

    * ``Column``
    * ``Series``
    * ``Index``
    * Scalars (can be broadcasted to a specified `length`)
    * Objects exposing ``__cuda_array_interface__`` (e.g., numba device arrays)
    * Objects exposing ``__array_interface__``(e.g., numpy arrays)
    * pyarrow array
    * pandas.Categorical objects
    """
    if isinstance(arbitrary, ColumnBase):
        if dtype is not None:
            return arbitrary.astype(dtype)
        else:
            return arbitrary

    elif isinstance(arbitrary, cudf.Series):
        data = arbitrary._column
        if dtype is not None:
            data = data.astype(dtype)
    elif isinstance(arbitrary, cudf.BaseIndex):
        data = arbitrary._values
        if dtype is not None:
            data = data.astype(dtype)

    elif type(arbitrary) is Buffer:
        if dtype is None:
            raise TypeError("dtype cannot be None if 'arbitrary' is a Buffer")

        data = build_column(arbitrary, dtype=dtype)

    elif hasattr(arbitrary, "__cuda_array_interface__"):
        desc = arbitrary.__cuda_array_interface__
        current_dtype = np.dtype(desc["typestr"])

        arb_dtype = check_cast_unsupported_dtype(current_dtype)

        if desc.get("mask", None) is not None:
            # Extract and remove the mask from arbitrary before
            # passing to cupy.asarray
            mask = _mask_from_cuda_array_interface_desc(arbitrary)
            arbitrary = SimpleNamespace(__cuda_array_interface__=desc.copy())
            arbitrary.__cuda_array_interface__["mask"] = None
            desc = arbitrary.__cuda_array_interface__
        else:
            mask = None

        arbitrary = cupy.asarray(arbitrary)

        if arb_dtype != current_dtype:
            arbitrary = arbitrary.astype(arb_dtype)
            current_dtype = arb_dtype

        if (
            desc["strides"] is not None
            and not (arbitrary.itemsize,) == arbitrary.strides
        ):
            arbitrary = cupy.ascontiguousarray(arbitrary)

        data = _data_from_cuda_array_interface_desc(arbitrary)
        col = build_column(data, dtype=current_dtype, mask=mask)

        if dtype is not None:
            col = col.astype(dtype)

        if isinstance(col, cudf.core.column.CategoricalColumn):
            return col
        elif np.issubdtype(col.dtype, np.floating):
            if nan_as_null or (mask is None and nan_as_null is None):
                mask = libcudf.transform.nans_to_nulls(col.fillna(np.nan))
                col = col.set_mask(mask)
        elif np.issubdtype(col.dtype, np.datetime64):
            if nan_as_null or (mask is None and nan_as_null is None):
                # Ignore typing error since this method is only defined for
                # DatetimeColumn, not the ColumnBase class.
                col = col._make_copy_with_na_as_null()  # type: ignore
        return col

    elif isinstance(arbitrary, (pa.Array, pa.ChunkedArray)):
        col = ColumnBase.from_arrow(arbitrary)
        if isinstance(arbitrary, pa.NullArray):
            if type(dtype) == str and dtype == "empty":
                new_dtype = pandas_dtype(arbitrary.type.to_pandas_dtype())
            else:
                new_dtype = pandas_dtype(dtype)
            col = col.astype(new_dtype)

        return col

    elif isinstance(arbitrary, (pd.Series, pd.Categorical)):
        if isinstance(arbitrary, pd.Series) and isinstance(
            arbitrary.array, pd.core.arrays.masked.BaseMaskedArray
        ):
            return as_column(arbitrary.array)
        if is_categorical_dtype(arbitrary):
            data = as_column(pa.array(arbitrary, from_pandas=True))
        elif is_interval_dtype(arbitrary.dtype):
            data = as_column(pa.array(arbitrary, from_pandas=True))
        elif arbitrary.dtype == np.bool_:
            data = as_column(cupy.asarray(arbitrary), dtype=arbitrary.dtype)
        elif arbitrary.dtype.kind in ("f"):
            arb_dtype = check_cast_unsupported_dtype(arbitrary.dtype)
            data = as_column(
                cupy.asarray(arbitrary, dtype=arb_dtype),
                nan_as_null=nan_as_null,
                dtype=dtype,
            )
        elif arbitrary.dtype.kind in ("u", "i"):
            data = as_column(
                cupy.asarray(arbitrary), nan_as_null=nan_as_null, dtype=dtype
            )
        else:
            pyarrow_array = pa.array(arbitrary, from_pandas=nan_as_null)
            if isinstance(pyarrow_array.type, pa.Decimal128Type):
                pyarrow_type = cudf.Decimal64Dtype.from_arrow(
                    pyarrow_array.type
                )
            else:
                pyarrow_type = arbitrary.dtype
            data = as_column(pyarrow_array, dtype=pyarrow_type)
        if dtype is not None:
            data = data.astype(dtype)

    elif isinstance(arbitrary, (pd.Timestamp, pd.Timedelta)):
        # This will always treat NaTs as nulls since it's not technically a
        # discrete value like NaN
        data = as_column(pa.array(pd.Series([arbitrary]), from_pandas=True))
        if dtype is not None:
            data = data.astype(dtype)

    elif np.isscalar(arbitrary) and not isinstance(arbitrary, memoryview):
        length = length or 1
        if (
            (nan_as_null is True)
            and isinstance(arbitrary, (np.floating, float))
            and np.isnan(arbitrary)
        ):
            arbitrary = None
            if dtype is None:
                dtype = np.dtype("float64")

        data = as_column(
            utils.scalar_broadcast_to(arbitrary, length, dtype=dtype)
        )
        if not nan_as_null and not is_decimal_dtype(data.dtype):
            if np.issubdtype(data.dtype, np.floating):
                data = data.fillna(np.nan)
            elif np.issubdtype(data.dtype, np.datetime64):
                data = data.fillna(np.datetime64("NaT"))

    elif hasattr(arbitrary, "__array_interface__"):
        # CUDF assumes values are always contiguous
        desc = arbitrary.__array_interface__
        shape = desc["shape"]
        arb_dtype = np.dtype(desc["typestr"])
        # CUDF assumes values are always contiguous
        if len(shape) > 1:
            raise ValueError("Data must be 1-dimensional")

        arbitrary = np.asarray(arbitrary)

        # Handle case that `arbitary` elements are cupy arrays
        if (
            shape
            and shape[0]
            and hasattr(arbitrary[0], "__cuda_array_interface__")
        ):
            return as_column(
                cupy.asarray(arbitrary, dtype=arbitrary[0].dtype),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )

        if not arbitrary.flags["C_CONTIGUOUS"]:
            arbitrary = np.ascontiguousarray(arbitrary)

        if dtype is not None:
            arbitrary = arbitrary.astype(dtype)

        if arb_dtype.kind == "M":

            time_unit = get_time_unit(arbitrary)
            cast_dtype = time_unit in ("D", "W", "M", "Y")

            if cast_dtype:
                arbitrary = arbitrary.astype(np.dtype("datetime64[s]"))

            buffer = Buffer(arbitrary.view("|u1"))
            mask = None
            if nan_as_null is None or nan_as_null is True:
                data = as_column(
                    buffer, dtype=arbitrary.dtype, nan_as_null=nan_as_null
                )
                data = data._make_copy_with_na_as_null()
                mask = data.mask

            data = cudf.core.column.datetime.DatetimeColumn(
                data=buffer, mask=mask, dtype=arbitrary.dtype
            )
        elif arb_dtype.kind == "m":

            time_unit = get_time_unit(arbitrary)
            cast_dtype = time_unit in ("D", "W", "M", "Y")

            if cast_dtype:
                arbitrary = arbitrary.astype(np.dtype("timedelta64[s]"))

            buffer = Buffer(arbitrary.view("|u1"))
            mask = None
            if nan_as_null is None or nan_as_null is True:
                data = as_column(
                    buffer, dtype=arbitrary.dtype, nan_as_null=nan_as_null
                )
                data = data._make_copy_with_na_as_null()
                mask = data.mask

            data = cudf.core.column.timedelta.TimeDeltaColumn(
                data=buffer,
                size=len(arbitrary),
                mask=mask,
                dtype=arbitrary.dtype,
            )
        elif (
            arbitrary.size != 0
            and arb_dtype.kind in ("O")
            and isinstance(arbitrary[0], pd._libs.interval.Interval)
        ):
            # changing from pd array to series,possible arrow bug
            interval_series = pd.Series(arbitrary)
            data = as_column(
                pa.Array.from_pandas(interval_series), dtype=arbitrary.dtype,
            )
            if dtype is not None:
                data = data.astype(dtype)
        elif arb_dtype.kind in ("O", "U"):
            data = as_column(
                pa.Array.from_pandas(arbitrary), dtype=arbitrary.dtype
            )
            # There is no cast operation available for pa.Array from int to
            # str, Hence instead of handling in pa.Array block, we
            # will have to type-cast here.
            if dtype is not None:
                data = data.astype(dtype)
        elif arb_dtype.kind in ("f"):
            arb_dtype = check_cast_unsupported_dtype(
                arb_dtype if dtype is None else dtype
            )
            data = as_column(
                cupy.asarray(arbitrary, dtype=arb_dtype),
                nan_as_null=nan_as_null,
            )
        else:
            data = as_column(cupy.asarray(arbitrary), nan_as_null=nan_as_null)

    elif isinstance(arbitrary, pd.core.arrays.numpy_.PandasArray):
        if is_categorical_dtype(arbitrary.dtype):
            arb_dtype = arbitrary.dtype
        else:
            if arbitrary.dtype == pd.StringDtype():
                arb_dtype = np.dtype("O")
            else:
                arb_dtype = check_cast_unsupported_dtype(arbitrary.dtype)
                if arb_dtype != arbitrary.dtype.numpy_dtype:
                    arbitrary = arbitrary.astype(arb_dtype)
        if (
            arbitrary.size != 0
            and isinstance(arbitrary[0], pd._libs.interval.Interval)
            and arb_dtype.kind in ("O")
        ):
            # changing from pd array to series,possible arrow bug
            interval_series = pd.Series(arbitrary)
            data = as_column(
                pa.Array.from_pandas(interval_series), dtype=arb_dtype
            )
        elif arb_dtype.kind in ("O", "U"):
            data = as_column(pa.Array.from_pandas(arbitrary), dtype=arb_dtype)
        else:
            data = as_column(
                pa.array(
                    arbitrary,
                    from_pandas=True if nan_as_null is None else nan_as_null,
                ),
                nan_as_null=nan_as_null,
            )
        if dtype is not None:
            data = data.astype(dtype)
    elif isinstance(arbitrary, memoryview):
        data = as_column(
            np.asarray(arbitrary), dtype=dtype, nan_as_null=nan_as_null
        )
    elif isinstance(arbitrary, cudf.Scalar):
        data = ColumnBase.from_scalar(arbitrary, length if length else 1)
    elif isinstance(arbitrary, pd.core.arrays.masked.BaseMaskedArray):
        cudf_dtype = arbitrary._data.dtype

        data = Buffer(arbitrary._data.view("|u1"))
        data = as_column(data, dtype=cudf_dtype)

        mask = arbitrary._mask
        mask = bools_to_mask(as_column(mask).unary_operator("not"))

        data = data.set_mask(mask)

    else:
        try:
            data = as_column(
                memoryview(arbitrary), dtype=dtype, nan_as_null=nan_as_null
            )
        except TypeError:
            pa_type = None
            np_type = None
            try:
                if dtype is not None:
                    if is_list_dtype(dtype):
                        data = pa.array(arbitrary)
                        if type(data) not in (pa.ListArray, pa.NullArray):
                            raise ValueError(
                                "Cannot create list column from given data"
                            )
                        return as_column(data, nan_as_null=nan_as_null)
                    if isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
                        data = pa.array(
                            arbitrary,
                            type=pa.decimal128(
                                precision=dtype.precision, scale=dtype.scale
                            ),
                        )
                        return cudf.core.column.Decimal64Column.from_arrow(
                            data
                        )
                    if isinstance(dtype, cudf.core.dtypes.Decimal32Dtype):
                        data = pa.array(
                            arbitrary,
                            type=pa.decimal128(
                                precision=dtype.precision, scale=dtype.scale
                            ),
                        )
                        return cudf.core.column.Decimal32Column.from_arrow(
                            data
                        )
                    dtype = pd.api.types.pandas_dtype(dtype)
                    if is_categorical_dtype(dtype) or is_interval_dtype(dtype):
                        raise TypeError
                    else:
                        np_type = np.dtype(dtype).type
                        if np_type == np.bool_:
                            pa_type = pa.bool_()
                        else:
                            pa_type = np_to_pa_dtype(np.dtype(dtype))
                data = as_column(
                    pa.array(
                        arbitrary,
                        type=pa_type,
                        from_pandas=True
                        if nan_as_null is None
                        else nan_as_null,
                    ),
                    dtype=dtype,
                    nan_as_null=nan_as_null,
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                if is_categorical_dtype(dtype):
                    sr = pd.Series(arbitrary, dtype="category")
                    data = as_column(sr, nan_as_null=nan_as_null, dtype=dtype)
                elif np_type == np.str_:
                    sr = pd.Series(arbitrary, dtype="str")
                    data = as_column(sr, nan_as_null=nan_as_null)
                elif is_interval_dtype(dtype):
                    sr = pd.Series(arbitrary, dtype="interval")
                    data = as_column(sr, nan_as_null=nan_as_null, dtype=dtype)
                else:
                    data = as_column(
                        _construct_array(arbitrary, dtype),
                        dtype=dtype,
                        nan_as_null=nan_as_null,
                    )
    return data


def _construct_array(
    arbitrary: Any, dtype: Optional[Dtype]
) -> Union[np.ndarray, cupy.ndarray]:
    """
    Construct a CuPy or NumPy array from `arbitrary`
    """
    try:
        dtype = dtype if dtype is None else np.dtype(dtype)
        arbitrary = cupy.asarray(arbitrary, dtype=dtype)
    except (TypeError, ValueError):
        native_dtype = dtype
        if (
            dtype is None
            and not cudf._lib.scalar._is_null_host_scalar(arbitrary)
            and infer_dtype(arbitrary) in ("mixed", "mixed-integer",)
        ):
            native_dtype = "object"
        arbitrary = np.asarray(
            arbitrary,
            dtype=native_dtype
            if native_dtype is None
            else np.dtype(native_dtype),
        )
    return arbitrary


def _data_from_cuda_array_interface_desc(obj) -> Buffer:
    desc = obj.__cuda_array_interface__
    ptr = desc["data"][0]
    nelem = desc["shape"][0] if len(desc["shape"]) > 0 else 1
    dtype = np.dtype(desc["typestr"])

    data = Buffer(data=ptr, size=nelem * dtype.itemsize, owner=obj)
    return data


def _mask_from_cuda_array_interface_desc(obj) -> Union[Buffer, None]:
    desc = obj.__cuda_array_interface__
    mask = desc.get("mask", None)

    if mask is not None:
        desc = mask.__cuda_array_interface__
        ptr = desc["data"][0]
        nelem = desc["shape"][0]
        typestr = desc["typestr"]
        typecode = typestr[1]
        if typecode == "t":
            mask_size = bitmask_allocation_size_bytes(nelem)
            mask = Buffer(data=ptr, size=mask_size, owner=obj)
        elif typecode == "b":
            col = as_column(mask)
            mask = bools_to_mask(col)
        else:
            raise NotImplementedError(
                f"Cannot infer mask from typestr {typestr}"
            )
    return mask


def serialize_columns(columns) -> Tuple[List[dict], List]:
    """
    Return the headers and frames resulting
    from serializing a list of Column
    Parameters
    ----------
    columns : list
        list of Columns to serialize
    Returns
    -------
    headers : list
        list of header metadata for each Column
    frames : list
        list of frames
    """
    headers: List[Dict[Any, Any]] = []
    frames = []

    if len(columns) > 0:
        header_columns = [c.serialize() for c in columns]
        headers, column_frames = zip(*header_columns)
        for f in column_frames:
            frames.extend(f)

    return headers, frames


def deserialize_columns(headers: List[dict], frames: List) -> List[ColumnBase]:
    """
    Construct a list of Columns from a list of headers
    and frames.
    """
    columns = []

    for meta in headers:
        col_frame_count = meta["frame_count"]
        col_typ = pickle.loads(meta["type-serialized"])
        colobj = col_typ.deserialize(meta, frames[:col_frame_count])
        columns.append(colobj)
        # Advance frames
        frames = frames[col_frame_count:]

    return columns


def arange(
    start: Union[int, float],
    stop: Union[int, float] = None,
    step: Union[int, float] = 1,
    dtype=None,
) -> ColumnBase:
    """
    Returns a column with evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop).
    The first three arguments are mapped like the range built-in function,
    i.e. start and step are optional.

    Parameters
    ----------
    start : int/float
        Start of the interval.
    stop : int/float, default is None
        Stop of the interval.
    step : int/float, default 1
        Step width between each pair of consecutive values.
    dtype : default None
        Data type specifier. It is inferred from other arguments by default.

    Returns
    -------
    cudf.core.column.NumericalColumn

    Examples
    --------
    >>> import cudf
    >>> col = cudf.core.column.arange(2, 7, 1, dtype='int16')
    >>> col
    <cudf.core.column.numerical.NumericalColumn object at 0x7ff7998f8b90>
    >>> cudf.Series(col)
    0    2
    1    3
    2    4
    3    5
    4    6
    dtype: int16
    """
    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    size = int(np.ceil((stop - start) / step))

    return libcudf.filling.sequence(
        size,
        as_device_scalar(start, dtype=dtype),
        as_device_scalar(step, dtype=dtype),
    )


def full(size: int, fill_value: ScalarLike, dtype: Dtype = None) -> ColumnBase:
    """
    Returns a column of given size and dtype, filled with a given value.

    Parameters
    ----------
    size : int
        size of the expected column.
    fill_value : scalar
         A scalar value to fill a new array.
    dtype : default None
        Data type specifier. It is inferred from other arguments by default.

    Returns
    -------
    Column

    Examples
    --------
    >>> import cudf
    >>> col = cudf.core.column.full(size=5, fill_value=7, dtype='int8')
    >>> col
    <cudf.core.column.numerical.NumericalColumn object at 0x7fa0912e8b90>
    >>> cudf.Series(col)
    0    7
    1    7
    2    7
    3    7
    4    7
    dtype: int8
    """
    return ColumnBase.from_scalar(cudf.Scalar(fill_value, dtype), size)


def _concat_columns(objs: "MutableSequence[ColumnBase]") -> ColumnBase:
    """Concatenate a sequence of columns."""
    if len(objs) == 0:
        dtype = pandas_dtype(None)
        return column_empty(0, dtype=dtype, masked=True)

    # If all columns are `NumericalColumn` with different dtypes,
    # we cast them to a common dtype.
    # Notice, we can always cast pure null columns
    not_null_col_dtypes = [o.dtype for o in objs if o.valid_count]
    if len(not_null_col_dtypes) and all(
        _is_non_decimal_numeric_dtype(dtyp)
        and np.issubdtype(dtyp, np.datetime64)
        for dtyp in not_null_col_dtypes
    ):
        # Use NumPy to find a common dtype
        common_dtype = np.find_common_type(not_null_col_dtypes, [])
        # Cast all columns to the common dtype
        objs = [obj.astype(common_dtype) for obj in objs]

    # Find the first non-null column:
    head = next((obj for obj in objs if obj.valid_count), objs[0])

    for i, obj in enumerate(objs):
        # Check that all columns are the same type:
        if not is_dtype_equal(obj.dtype, head.dtype):
            # if all null, cast to appropriate dtype
            if obj.valid_count == 0:
                objs[i] = column_empty_like(
                    head, dtype=head.dtype, masked=True, newsize=len(obj)
                )
            else:
                raise ValueError("All columns must be the same type")

    # TODO: This logic should be generalized to a dispatch to
    # ColumnBase._concat so that all subclasses can override necessary
    # behavior. However, at the moment it's not clear what that API should look
    # like, so CategoricalColumn simply implements a minimal working API.
    if all(is_categorical_dtype(o.dtype) for o in objs):
        return cudf.core.column.categorical.CategoricalColumn._concat(
            cast(
                MutableSequence[
                    cudf.core.column.categorical.CategoricalColumn
                ],
                objs,
            )
        )

    newsize = sum(map(len, objs))
    if newsize > libcudf.MAX_COLUMN_SIZE:
        raise MemoryError(
            f"Result of concat cannot have "
            f"size > {libcudf.MAX_COLUMN_SIZE_STR}"
        )
    elif newsize == 0:
        col = column_empty(0, head.dtype, masked=True)
    else:
        # Filter out inputs that have 0 length, then concatenate.
        objs = [o for o in objs if len(o)]
        try:
            col = libcudf.concat.concat_columns(objs)
        except RuntimeError as e:
            if "concatenated rows exceeds size_type range" in str(e):
                raise OverflowError(
                    "total size of output is too large for a cudf column"
                ) from e
            raise
    return col
