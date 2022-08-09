# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from __future__ import annotations

import pickle
import warnings
from functools import cached_property
from itertools import chain
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

import rmm

import cudf
from cudf import _lib as libcudf
from cudf._lib.column import Column
from cudf._lib.null_mask import (
    MaskState,
    bitmask_allocation_size_bytes,
    create_null_mask,
)
from cudf._lib.scalar import as_device_scalar
from cudf._lib.stream_compaction import (
    apply_boolean_mask,
    distinct_count as cpp_distinct_count,
    drop_duplicates,
    drop_nulls,
)
from cudf._lib.transform import bools_to_mask
from cudf._typing import ColumnLike, Dtype, ScalarLike
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    infer_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_decimal32_dtype,
    is_decimal64_dtype,
    is_decimal128_dtype,
    is_decimal_dtype,
    is_dtype_equal,
    is_integer_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_scalar,
    is_string_dtype,
    is_struct_dtype,
)
from cudf.core.abc import Serializable
from cudf.core.buffer import Buffer, DeviceBufferLike, as_device_buffer_like
from cudf.core.dtypes import (
    CategoricalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.missing import NA
from cudf.core.mixins import BinaryOperand, Reducible
from cudf.utils.dtypes import (
    _maybe_convert_to_default_type,
    cudf_dtype_from_pa_type,
    get_time_unit,
    min_unsigned_type,
    np_to_pa_dtype,
    pandas_dtypes_alias_to_cudf_alias,
    pandas_dtypes_to_np_dtypes,
)
from cudf.utils.utils import _array_ufunc, mask_dtype

T = TypeVar("T", bound="ColumnBase")
# TODO: This workaround allows type hints for `slice`, since `slice` is a
# method in ColumnBase.
Slice = TypeVar("Slice", bound=slice)


class ColumnBase(Column, Serializable, BinaryOperand, Reducible):
    _VALID_REDUCTIONS = {
        "any",
        "all",
        "max",
        "min",
    }

    def as_frame(self) -> "cudf.core.frame.Frame":
        """
        Converts a Column to Frame
        """
        return cudf.core.single_column_frame.SingleColumnFrame(
            {None: self.copy(deep=False)}
        )

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

    @property
    def values_host(self) -> "np.ndarray":
        """
        Return a numpy representation of the Column.
        """
        if len(self) == 0:
            return np.array([], dtype=self.dtype)

        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        return self.data_array_view.copy_to_host()

    @property
    def values(self) -> "cupy.ndarray":
        """
        Return a CuPy representation of the Column.
        """
        if len(self) == 0:
            return cupy.array([], dtype=self.dtype)

        if self.has_nulls():
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
        if check_dtypes and (self.dtype != other.dtype):
            return False
        ret = self._binaryop(other, "NULL_EQUALS")
        if ret is NotImplemented:
            raise TypeError(f"Cannot compare equality with {type(other)}")
        return ret.all()

    def all(self, skipna: bool = True) -> bool:
        # The skipna argument is only used for numerical columns.
        # If all entries are null the result is True, including when the column
        # is empty.

        if self.null_count == self.size:
            return True

        return libcudf.reduce.reduce("all", self, dtype=np.bool_)

    def any(self, skipna: bool = True) -> bool:
        # Early exit for fast cases.

        if not skipna and self.has_nulls():
            return True
        elif skipna and self.null_count == self.size:
            return False

        return libcudf.reduce.reduce("any", self, dtype=np.bool_)

    def dropna(self, drop_nan: bool = False) -> ColumnBase:
        # The drop_nan argument is only used for numerical columns.
        return drop_nulls([self])[0]

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
        return libcudf.interop.to_arrow([self], [["None"]],)[
            "None"
        ].chunk(0)

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

            codes = libcudf.interop.from_arrow(indices_table)[0]
            categories = libcudf.interop.from_arrow(dictionaries_table)[0]

            return build_categorical_column(
                categories=categories,
                codes=codes,
                mask=codes.base_mask,
                size=codes.size,
                ordered=array.type.ordered,
            )
        elif isinstance(
            array.type, pd.core.arrays._arrow_utils.ArrowIntervalType
        ):
            return cudf.core.column.IntervalColumn.from_arrow(array)

        result = libcudf.interop.from_arrow(data)[0]

        return result._with_type_metadata(cudf_dtype_from_pa_type(array.type))

    def _get_mask_as_column(self) -> ColumnBase:
        return libcudf.transform.mask_to_bools(
            self.base_mask, self.offset, self.offset + len(self)
        )

    @cached_property
    def memory_usage(self) -> int:
        n = 0
        if self.data is not None:
            n += self.data.size
        if self.nullable:
            n += bitmask_allocation_size_bytes(self.size)
        return n

    def _fill(
        self,
        fill_value: ScalarLike,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Optional[ColumnBase]:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        # Constructing a cuDF scalar can cut unnecessary DtoH copy if
        # the scalar is None when calling `is_valid`.
        slr = cudf.Scalar(fill_value, dtype=self.dtype)

        if not inplace:
            return libcudf.filling.fill(self, begin, end, slr.device_value)

        if is_string_dtype(self.dtype):
            return self._mimic_inplace(
                libcudf.filling.fill(self, begin, end, slr.device_value),
                inplace=True,
            )

        if not slr.is_valid() and not self.nullable:
            mask = create_null_mask(self.size, state=MaskState.ALL_VALID)
            self.set_base_mask(mask)

        libcudf.filling.fill_in_place(self, begin, end, slr.device_value)

        return self

    def shift(self, offset: int, fill_value: ScalarLike) -> ColumnBase:
        return libcudf.copying.shift(self, offset, fill_value)

    @property
    def valid_count(self) -> int:
        """Number of non-null values"""
        return len(self) - self.null_count

    @property
    def nullmask(self) -> DeviceBufferLike:
        """The gpu buffer for the null-mask"""
        if not self.nullable:
            raise ValueError("Column has no null mask")
        return self.mask_array_view

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

        dtype = cudf.dtype(dtype)

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

            # This assertion prevents mypy errors below.
            assert self.base_data is not None

            start = self.offset * self.dtype.itemsize
            end = start + self.size * self.dtype.itemsize
            return build_column(self.base_data[start:end], dtype=dtype)

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
            return libcudf.copying.column_slice(self, [start, stop])[
                0
            ]._with_type_metadata(self.dtype)
        else:
            # Need to create a gather map for given slice with stride
            gather_map = arange(
                start=start,
                stop=stop,
                step=stride,
                dtype=cudf.dtype(np.int32),
            )
            return self.take(gather_map)

    def __setitem__(self, key: Any, value: Any):
        """
        Set the value of ``self[key]`` to ``value``.

        If ``value`` and ``self`` are of different types, ``value`` is coerced
        to ``self.dtype``. Assumes ``self`` and ``value`` are index-aligned.
        """

        # Normalize value to scalar/column
        value_normalized = (
            cudf.Scalar(value, dtype=self.dtype)
            if is_scalar(value)
            else as_column(value, dtype=self.dtype)
        )

        out: Optional[ColumnBase]  # If None, no need to perform mimic inplace.
        if isinstance(key, slice):
            out = self._scatter_by_slice(key, value_normalized)
        else:
            key = as_column(key)
            if not isinstance(key, cudf.core.column.NumericalColumn):
                raise ValueError(f"Invalid scatter map type {key.dtype}.")
            out = self._scatter_by_column(key, value_normalized)

        if out:
            self._mimic_inplace(out, inplace=True)

    def _wrap_binop_normalization(self, other):
        if other is NA or other is None:
            return cudf.Scalar(other, dtype=self.dtype)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            other = other.item()
        return self.normalize_binop_value(other)

    def _scatter_by_slice(
        self, key: Slice, value: Union[cudf.core.scalar.Scalar, ColumnBase]
    ) -> Optional[ColumnBase]:
        """If this function returns None, it's either a no-op (slice is empty),
        or the inplace replacement is already performed (fill-in-place).
        """
        start, stop, step = key.indices(len(self))
        if start >= stop:
            return None
        num_keys = (stop - start) // step

        self._check_scatter_key_length(num_keys, value)

        if step == 1:
            if isinstance(value, cudf.core.scalar.Scalar):
                return self._fill(value, start, stop, inplace=True)
            else:
                return libcudf.copying.copy_range(
                    value, self, 0, num_keys, start, stop, False
                )

        # step != 1, create a scatter map with arange
        scatter_map = arange(
            start=start,
            stop=stop,
            step=step,
            dtype=cudf.dtype(np.int32),
        )

        return self._scatter_by_column(scatter_map, value)

    def _scatter_by_column(
        self,
        key: cudf.core.column.NumericalColumn,
        value: Union[cudf.core.scalar.Scalar, ColumnBase],
    ) -> ColumnBase:
        if is_bool_dtype(key.dtype):
            # `key` is boolean mask
            if len(key) != len(self):
                raise ValueError(
                    "Boolean mask must be of same length as column"
                )
            if isinstance(value, ColumnBase) and len(self) == len(value):
                # Both value and key are aligned to self. Thus, the values
                # corresponding to the false values in key should be
                # ignored.
                value = value.apply_boolean_mask(key)
                # After applying boolean mask, the length of value equals
                # the number of elements to scatter, we can skip computing
                # the sum of ``key`` below.
                num_keys = len(value)
            else:
                # Compute the number of element to scatter by summing all
                # `True`s in the boolean mask.
                num_keys = key.sum()
        else:
            # `key` is integer scatter map
            num_keys = len(key)

        self._check_scatter_key_length(num_keys, value)

        try:
            if is_bool_dtype(key.dtype):
                return libcudf.copying.boolean_mask_scatter(
                    [value], [self], key
                )[0]._with_type_metadata(self.dtype)
            else:
                return libcudf.copying.scatter([value], key, [self])[
                    0
                ]._with_type_metadata(self.dtype)
        except RuntimeError as e:
            if "out of bounds" in str(e):
                raise IndexError(
                    f"index out of bounds for column of size {len(self)}"
                ) from e
            raise

    def _check_scatter_key_length(
        self, num_keys: int, value: Union[cudf.core.scalar.Scalar, ColumnBase]
    ):
        """`num_keys` is the number of keys to scatter. Should equal to the
        number of rows in ``value`` if ``value`` is a column.
        """
        if isinstance(value, ColumnBase):
            if len(value) != num_keys:
                msg = (
                    f"Size mismatch: cannot set value "
                    f"of size {len(value)} to indexing result of size "
                    f"{num_keys}"
                )
                raise ValueError(msg)

    def fillna(
        self: T,
        value: Any = None,
        method: str = None,
        dtype: Dtype = None,
    ) -> T:
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        return libcudf.replace.replace_nulls(
            input_col=self, replacement=value, method=method, dtype=dtype
        )

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column."""
        result = libcudf.unary.is_null(self)

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values incase
            # of a float column
            result = result | libcudf.unary.is_nan(self)

        return result

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column."""
        result = libcudf.unary.is_valid(self)

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values incase
            # of a float column
            result = result & libcudf.unary.is_non_nan(self)

        return result

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
        return concat_columns([self, as_column(other)])

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase:
        raise TypeError(f"cannot perform quantile with type {self.dtype}")

    def take(
        self: T, indices: ColumnBase, nullify: bool = False, check_bounds=True
    ) -> T:
        """Return Column by taking values from the corresponding *indices*.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        # Handle zero size
        if indices.size == 0:
            return cast(T, column_empty_like(self, newsize=0))

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if not is_integer_dtype(indices.dtype):
            indices = indices.astype("int32")
        if not libcudf.copying._gather_map_is_valid(
            indices, len(self), check_bounds, nullify
        ):
            raise IndexError("Gather map index is out of bounds.")

        return libcudf.copying.gather([self], indices, nullify=nullify)[
            0
        ]._with_type_metadata(self.dtype)

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

        return lhs._obtain_isin_result(rhs)

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
                return self.isnull()
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
        return res._data["bool"].fillna(False)

    def as_mask(self) -> DeviceBufferLike:
        """Convert booleans to bitmask

        Returns
        -------
        DeviceBufferLike
        """

        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        return bools_to_mask(self)

    @property
    def is_unique(self) -> bool:
        return self.distinct_count() == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        return not self.has_nulls() and self.as_frame()._is_sorted(
            ascending=None, null_position=None
        )

    @property
    def is_monotonic_decreasing(self) -> bool:
        return not self.has_nulls() and self.as_frame()._is_sorted(
            ascending=[False], null_position=None
        )

    def get_slice_bound(self, label: ScalarLike, side: str, kind: str) -> int:
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
        if kind not in {"ix", "loc", "getitem", None}:
            raise ValueError(
                f"Invalid value for ``kind`` parameter,"
                f" must be either one of the following: "
                f"{'ix', 'loc', 'getitem', None}, but found: {kind}"
            )
        if side not in {"left", "right"}:
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
        na_position: str = "last",
    ) -> Tuple[ColumnBase, "cudf.core.column.NumericalColumn"]:
        col_inds = self.as_frame()._get_sorted_inds(
            ascending=ascending, na_position=na_position
        )
        col_keys = self.take(col_inds)
        return col_keys, col_inds

    def distinct_count(self, dropna: bool = True) -> int:
        try:
            return self._distinct_count[dropna]
        except KeyError:
            self._distinct_count[dropna] = cpp_distinct_count(
                self, ignore_nulls=dropna
            )
            return self._distinct_count[dropna]

    def can_cast_safely(self, to_dtype: Dtype) -> bool:
        raise NotImplementedError()

    def astype(self, dtype: Dtype, **kwargs) -> ColumnBase:
        if is_categorical_dtype(dtype):
            return self.as_categorical_column(dtype, **kwargs)

        dtype = (
            pandas_dtypes_alias_to_cudf_alias.get(dtype, dtype)
            if isinstance(dtype, str)
            else pandas_dtypes_to_np_dtypes.get(dtype, dtype)
        )
        if _is_non_decimal_numeric_dtype(dtype):
            return self.as_numerical_column(dtype, **kwargs)
        elif is_categorical_dtype(dtype):
            return self.as_categorical_column(dtype, **kwargs)
        elif cudf.dtype(dtype).type in {
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
        elif np.issubdtype(cast(Any, dtype), np.datetime64):
            return self.as_datetime_column(dtype, **kwargs)
        elif np.issubdtype(cast(Any, dtype), np.timedelta64):
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
        if (
            isinstance(dtype, cudf.CategoricalDtype)
            and dtype._categories is not None
        ) or (
            isinstance(dtype, pd.CategoricalDtype)
            and dtype.categories is not None
        ):
            labels = sr._label_encoding(cats=dtype.categories)
            if "ordered" in kwargs:
                warnings.warn(
                    "Ignoring the `ordered` parameter passed in `**kwargs`, "
                    "will be using `ordered` parameter of CategoricalDtype"
                )

            return build_categorical_column(
                categories=as_column(dtype.categories),
                codes=labels._column,
                mask=self.mask,
                ordered=dtype.ordered,
            )

        cats = sr.unique().astype(sr.dtype)
        label_dtype = min_unsigned_type(len(cats))
        labels = sr._label_encoding(
            cats=cats, dtype=label_dtype, na_sentinel=1
        )

        # columns include null index in factorization; remove:
        if self.has_nulls():
            cats = cats._column.dropna(drop_nan=False)
            min_type = min_unsigned_type(len(cats), 8)
            labels = labels - 1
            if cudf.dtype(min_type).itemsize < labels.dtype.itemsize:
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
    ) -> Union["cudf.core.column.decimal.DecimalBaseColumn"]:
        raise NotImplementedError

    def as_decimal128_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.Decimal128Column":
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
        mask = as_column(mask)
        if not is_bool_dtype(mask.dtype):
            raise ValueError("boolean_mask is not boolean type.")

        return apply_boolean_mask([self], mask)[0]._with_type_metadata(
            self.dtype
        )

    def argsort(
        self, ascending: bool = True, na_position: str = "last"
    ) -> ColumnBase:

        return self.as_frame()._get_sorted_inds(
            ascending=ascending, na_position=na_position
        )

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            f"dtype {self.dtype} is not yet supported via "
            "`__cuda_array_interface__`"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    def searchsorted(
        self,
        value,
        side: str = "left",
        ascending: bool = True,
        na_position: str = "last",
    ):
        values = as_column(value).as_frame()
        return self.as_frame().searchsorted(
            values, side, ascending=ascending, na_position=na_position
        )

    def unique(self) -> ColumnBase:
        """
        Get unique values in the data
        """
        # TODO: We could avoid performing `drop_duplicates` for
        # columns with values that already are unique.
        # Few things to note before we can do this optimization is
        # the following issue resolved:
        # https://github.com/rapidsai/cudf/issues/5286

        return drop_duplicates([self], keep="first")[0]

    def serialize(self) -> Tuple[dict, list]:
        # data model:

        # Serialization produces a nested metadata "header" and a flattened
        # list of memoryviews/buffers that reference data (frames).  Each
        # header advertises a frame_count slot which indicates how many
        # frames deserialization will consume. The class used to construct
        # an object is named under the key "type-serialized" to match with
        # Dask's serialization protocol (see
        # distributed.protocol.serialize). Since column dtypes may either be
        # cudf native or foreign some special-casing is required here for
        # serialization.

        header: Dict[Any, Any] = {}
        frames = []
        header["type-serialized"] = pickle.dumps(type(self))
        try:
            dtype, dtype_frames = self.dtype.serialize()
            header["dtype"] = dtype
            frames.extend(dtype_frames)
            header["dtype-is-cudf-serialized"] = True
        except AttributeError:
            header["dtype"] = pickle.dumps(self.dtype)
            header["dtype-is-cudf-serialized"] = False

        if self.data is not None:
            data_header, data_frames = self.data.serialize()
            header["data"] = data_header
            frames.extend(data_frames)

        if self.mask is not None:
            mask_header, mask_frames = self.mask.serialize()
            header["mask"] = mask_header
            frames.extend(mask_frames)
        if self.children:
            child_headers, child_frames = zip(
                *(c.serialize() for c in self.children)
            )
            header["subheaders"] = list(child_headers)
            frames.extend(chain(*child_frames))
        header["size"] = self.size
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> ColumnBase:
        def unpack(header, frames) -> Tuple[Any, list]:
            count = header["frame_count"]
            klass = pickle.loads(header["type-serialized"])
            obj = klass.deserialize(header, frames[:count])
            return obj, frames[count:]

        assert header["frame_count"] == len(frames), (
            f"Deserialization expected {header['frame_count']} frames, "
            f"but received {len(frames)}"
        )
        if header["dtype-is-cudf-serialized"]:
            dtype, frames = unpack(header["dtype"], frames)
        else:
            dtype = pickle.loads(header["dtype"])
        if "data" in header:
            data, frames = unpack(header["data"], frames)
        else:
            data = None
        if "mask" in header:
            mask, frames = unpack(header["mask"], frames)
        else:
            mask = None
        children = []
        if "subheaders" in header:
            for h in header["subheaders"]:
                child, frames = unpack(h, frames)
                children.append(child)
        assert len(frames) == 0, "Deserialization did not consume all frames"
        return build_column(
            data=data,
            dtype=dtype,
            mask=mask,
            size=header.get("size", None),
            children=tuple(children),
        )

    def unary_operator(self, unaryop: str):
        raise TypeError(
            f"Operation {unaryop} not supported for dtype {self.dtype}."
        )

    def normalize_binop_value(
        self, other: ScalarLike
    ) -> Union[ColumnBase, ScalarLike]:
        raise NotImplementedError

    def _minmax(self, skipna: bool = None):
        result_col = self._process_for_reduction(skipna=skipna)
        if isinstance(result_col, ColumnBase):
            return libcudf.reduce.minmax(result_col)
        return result_col

    def _reduce(
        self, op: str, skipna: bool = None, min_count: int = 0, *args, **kwargs
    ) -> ScalarLike:
        """Compute {op} of column values.

        skipna : bool
            Whether or not na values must be skipped.
        min_count : int, default 0
            The minimum number of entries for the reduction, otherwise the
            reduction returns NaN.
        """
        preprocessed = self._process_for_reduction(
            skipna=skipna, min_count=min_count
        )
        if isinstance(preprocessed, ColumnBase):
            return libcudf.reduce.reduce(op, preprocessed, **kwargs)
        return preprocessed

    @property
    def contains_na_entries(self) -> bool:
        return self.null_count != 0

    def _process_for_reduction(
        self, skipna: bool = None, min_count: int = 0
    ) -> Union[ColumnBase, ScalarLike]:
        skipna = True if skipna is None else skipna

        if skipna:
            if self.has_nulls():
                result_col = self.dropna()
        else:
            if self.has_nulls():
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

    def _reduction_result_dtype(self, reduction_op: str) -> Dtype:
        """
        Determine the correct dtype to pass to libcudf based on
        the input dtype, data dtype, and specific reduction op
        """
        return self.dtype

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
    """Allocate a new column like the given *column*"""
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
    """Allocate a new column like the given row_count and dtype."""
    dtype = cudf.dtype(dtype)
    children = ()  # type: Tuple[ColumnBase, ...]

    if is_struct_dtype(dtype):
        data = None
        children = tuple(
            column_empty(row_count, field_dtype)
            for field_dtype in dtype.fields.values()
        )
    elif is_list_dtype(dtype):
        data = None
        children = (
            full(row_count + 1, 0, dtype="int32"),
            column_empty(row_count, dtype=dtype.element_type),
        )
    elif is_categorical_dtype(dtype):
        data = None
        children = (
            build_column(
                data=as_device_buffer_like(
                    rmm.DeviceBuffer(
                        size=row_count * cudf.dtype("int32").itemsize
                    )
                ),
                dtype="int32",
            ),
        )
    elif dtype.kind in "OU" and not is_decimal_dtype(dtype):
        data = None
        children = (
            full(row_count + 1, 0, dtype="int32"),
            build_column(
                data=as_device_buffer_like(
                    rmm.DeviceBuffer(
                        size=row_count * cudf.dtype("int8").itemsize
                    )
                ),
                dtype="int8",
            ),
        )
    else:
        data = as_device_buffer_like(
            rmm.DeviceBuffer(size=row_count * dtype.itemsize)
        )

    if masked:
        mask = create_null_mask(row_count, state=MaskState.ALL_NULL)
    else:
        mask = None

    return build_column(
        data, dtype, mask=mask, size=row_count, children=children
    )


def build_column(
    data: Union[DeviceBufferLike, None],
    dtype: Dtype,
    *,
    size: int = None,
    mask: DeviceBufferLike = None,
    offset: int = 0,
    null_count: int = None,
    children: Tuple[ColumnBase, ...] = (),
) -> ColumnBase:
    """
    Build a Column of the appropriate type from the given parameters

    Parameters
    ----------
    data : DeviceBufferLike
        The data buffer (can be None if constructing certain Column
        types like StringColumn, ListColumn, or CategoricalColumn)
    dtype
        The dtype associated with the Column to construct
    mask : DeviceBufferLike, optional
        The mask buffer
    size : int, optional
    offset : int, optional
    children : tuple, optional
    """
    dtype = cudf.dtype(dtype)

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
            offset=offset,
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
    elif is_decimal128_dtype(dtype):
        if size is None:
            raise TypeError("Must specify size")
        return cudf.core.column.Decimal128Column(
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
    mask: DeviceBufferLike = None,
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
    mask : DeviceBufferLike
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
    mask : DeviceBufferLike
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
    mask: DeviceBufferLike = None,
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
    mask: DeviceBufferLike
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
    mask: DeviceBufferLike = None,
    size: int = None,
    offset: int = 0,
    null_count: int = None,
) -> "cudf.core.column.StructColumn":
    """
    Build a StructColumn

    Parameters
    ----------
    names : sequence of strings
        Field names to map to children dtypes, must be strings.
    children : tuple

    mask: DeviceBufferLike
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


def _make_copy_replacing_NaT_with_null(column):
    """Return a copy with NaT values replaced with nulls."""
    if np.issubdtype(column.dtype, np.timedelta64):
        na_value = np.timedelta64("NaT", column.time_unit)
    elif np.issubdtype(column.dtype, np.datetime64):
        na_value = np.datetime64("NaT", column.time_unit)
    else:
        raise ValueError("This type does not support replacing NaT with null.")

    null = column_empty_like(column, masked=True, newsize=1)
    out_col = cudf._lib.replace.replace(
        column,
        build_column(
            as_device_buffer_like(
                np.array([na_value], dtype=column.dtype).view("|u1")
            ),
            dtype=column.dtype,
        ),
        null,
    )
    return out_col


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

    elif hasattr(arbitrary, "__cuda_array_interface__"):
        desc = arbitrary.__cuda_array_interface__
        current_dtype = np.dtype(desc["typestr"])

        arb_dtype = (
            np.dtype("float32")
            if current_dtype == "float16"
            else cudf.dtype(current_dtype)
        )

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

        data = as_device_buffer_like(arbitrary)
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
                col = _make_copy_replacing_NaT_with_null(col)
        return col

    elif isinstance(arbitrary, (pa.Array, pa.ChunkedArray)):
        if isinstance(arbitrary, pa.lib.HalfFloatArray):
            raise NotImplementedError(
                "Type casting from `float16` to `float32` is not "
                "yet supported in pyarrow, see: "
                "https://issues.apache.org/jira/browse/ARROW-3802"
            )
        col = ColumnBase.from_arrow(arbitrary)

        if isinstance(arbitrary, pa.NullArray):
            new_dtype = cudf.dtype(arbitrary.type.to_pandas_dtype())
            if dtype is not None:
                # Cast the column to the `dtype` if specified.
                col = col.astype(dtype)
            elif len(arbitrary) == 0:
                # If the column is empty, it has to be
                # a `float64` dtype.
                col = col.astype("float64")
            else:
                # If the null column is not empty, it has to
                # be of `object` dtype.
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
            arb_dtype = np.dtype(arbitrary.dtype)
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
                pyarrow_type = cudf.Decimal128Dtype.from_arrow(
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
                dtype = cudf.dtype("float64")

        data = as_column(full(length, arbitrary, dtype=dtype))
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

        # Handle case that `arbitrary` elements are cupy arrays
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

        delayed_cast = False
        if dtype is not None:
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                # Some `dtype`'s can't be parsed by `np.dtype`
                # for which we will have to cast after the column
                # has been constructed.
                delayed_cast = True
            else:
                arbitrary = arbitrary.astype(dtype)

        if arb_dtype.kind == "M":

            time_unit = get_time_unit(arbitrary)
            cast_dtype = time_unit in ("D", "W", "M", "Y")

            if cast_dtype:
                arbitrary = arbitrary.astype(cudf.dtype("datetime64[s]"))

            buffer = as_device_buffer_like(arbitrary.view("|u1"))
            mask = None
            if nan_as_null is None or nan_as_null is True:
                data = build_column(buffer, dtype=arbitrary.dtype)
                data = _make_copy_replacing_NaT_with_null(data)
                mask = data.mask

            data = cudf.core.column.datetime.DatetimeColumn(
                data=buffer, mask=mask, dtype=arbitrary.dtype
            )
        elif arb_dtype.kind == "m":

            time_unit = get_time_unit(arbitrary)
            cast_dtype = time_unit in ("D", "W", "M", "Y")

            if cast_dtype:
                arbitrary = arbitrary.astype(cudf.dtype("timedelta64[s]"))

            buffer = as_device_buffer_like(arbitrary.view("|u1"))
            mask = None
            if nan_as_null is None or nan_as_null is True:
                data = build_column(buffer, dtype=arbitrary.dtype)
                data = _make_copy_replacing_NaT_with_null(data)
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
                pa.Array.from_pandas(interval_series),
                dtype=arbitrary.dtype,
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
            if arb_dtype == np.dtype("float16"):
                arb_dtype = np.dtype("float32")
            arb_dtype = cudf.dtype(arb_dtype if dtype is None else dtype)
            data = as_column(
                cupy.asarray(arbitrary, dtype=arb_dtype),
                nan_as_null=nan_as_null,
            )
        else:
            data = as_column(cupy.asarray(arbitrary), nan_as_null=nan_as_null)

        if delayed_cast:
            data = data.astype(cudf.dtype(dtype))

    elif isinstance(arbitrary, pd.core.arrays.numpy_.PandasArray):
        if is_categorical_dtype(arbitrary.dtype):
            arb_dtype = arbitrary.dtype
        else:
            if arbitrary.dtype == pd.StringDtype():
                arb_dtype = cudf.dtype("O")
            else:
                arb_dtype = (
                    cudf.dtype("float32")
                    if arbitrary.dtype == "float16"
                    else cudf.dtype(arbitrary.dtype)
                )
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
        data = as_column(pa.Array.from_pandas(arbitrary), dtype=dtype)
    else:
        try:
            data = as_column(
                memoryview(arbitrary), dtype=dtype, nan_as_null=nan_as_null
            )
        except TypeError:
            if dtype is not None:
                # Arrow throws a type error if the input is of
                # mixed-precision and cannot fit into the provided
                # decimal type properly, see:
                # https://github.com/apache/arrow/pull/9948
                # Hence we should let the exception propagate to
                # the user.
                if isinstance(dtype, cudf.core.dtypes.Decimal128Dtype):
                    data = pa.array(
                        arbitrary,
                        type=pa.decimal128(
                            precision=dtype.precision, scale=dtype.scale
                        ),
                    )
                    return cudf.core.column.Decimal128Column.from_arrow(data)
                elif isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
                    data = pa.array(
                        arbitrary,
                        type=pa.decimal128(
                            precision=dtype.precision, scale=dtype.scale
                        ),
                    )
                    return cudf.core.column.Decimal64Column.from_arrow(data)
                elif isinstance(dtype, cudf.core.dtypes.Decimal32Dtype):
                    data = pa.array(
                        arbitrary,
                        type=pa.decimal128(
                            precision=dtype.precision, scale=dtype.scale
                        ),
                    )
                    return cudf.core.column.Decimal32Column.from_arrow(data)

            pa_type = None
            np_type = None
            try:
                if dtype is not None:
                    if is_categorical_dtype(dtype) or is_interval_dtype(dtype):
                        raise TypeError
                    if is_list_dtype(dtype):
                        data = pa.array(arbitrary)
                        if type(data) not in (pa.ListArray, pa.NullArray):
                            raise ValueError(
                                "Cannot create list column from given data"
                            )
                        return as_column(data, nan_as_null=nan_as_null)
                    elif isinstance(
                        dtype, cudf.StructDtype
                    ) and not isinstance(dtype, cudf.IntervalDtype):
                        data = pa.array(arbitrary, type=dtype.to_arrow())
                        return as_column(data, nan_as_null=nan_as_null)
                    elif isinstance(dtype, cudf.core.dtypes.Decimal128Dtype):
                        data = pa.array(
                            arbitrary,
                            type=pa.decimal128(
                                precision=dtype.precision, scale=dtype.scale
                            ),
                        )
                        return cudf.core.column.Decimal128Column.from_arrow(
                            data
                        )
                    elif isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
                        data = pa.array(
                            arbitrary,
                            type=pa.decimal128(
                                precision=dtype.precision, scale=dtype.scale
                            ),
                        )
                        return cudf.core.column.Decimal64Column.from_arrow(
                            data
                        )
                    elif isinstance(dtype, cudf.core.dtypes.Decimal32Dtype):
                        data = pa.array(
                            arbitrary,
                            type=pa.decimal128(
                                precision=dtype.precision, scale=dtype.scale
                            ),
                        )
                        return cudf.core.column.Decimal32Column.from_arrow(
                            data
                        )

                    if is_bool_dtype(dtype):
                        # Need this special case handling for bool dtypes,
                        # since 'boolean' & 'pd.BooleanDtype' are not
                        # understood by np.dtype below.
                        dtype = "bool"
                    np_type = np.dtype(dtype).type
                    pa_type = np_to_pa_dtype(np.dtype(dtype))
                else:
                    # By default cudf constructs a 64-bit column. Setting
                    # the `default_*_bitwidth` to 32 will result in a 32-bit
                    # column being created.
                    if (
                        cudf.get_option("default_integer_bitwidth")
                        and infer_dtype(arbitrary) == "integer"
                    ):
                        pa_type = np_to_pa_dtype(
                            _maybe_convert_to_default_type("int")
                        )
                    if cudf.get_option(
                        "default_float_bitwidth"
                    ) and infer_dtype(arbitrary) in (
                        "floating",
                        "mixed-integer-float",
                    ):
                        pa_type = np_to_pa_dtype(
                            _maybe_convert_to_default_type("float")
                        )

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
                elif (
                    isinstance(arbitrary, Sequence)
                    and len(arbitrary) > 0
                    and any(
                        cudf.utils.dtypes.is_column_like(arb)
                        for arb in arbitrary
                    )
                ):
                    return cudf.core.column.ListColumn.from_sequences(
                        arbitrary
                    )
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
        dtype = dtype if dtype is None else cudf.dtype(dtype)
        arbitrary = cupy.asarray(arbitrary, dtype=dtype)
    except (TypeError, ValueError):
        native_dtype = dtype
        if (
            dtype is None
            and not cudf._lib.scalar._is_null_host_scalar(arbitrary)
            and infer_dtype(arbitrary)
            in (
                "mixed",
                "mixed-integer",
            )
        ):
            native_dtype = "object"
        arbitrary = np.asarray(
            arbitrary,
            dtype=native_dtype
            if native_dtype is None
            else np.dtype(native_dtype),
        )
    return arbitrary


def _mask_from_cuda_array_interface_desc(obj) -> Union[DeviceBufferLike, None]:
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
) -> cudf.core.column.NumericalColumn:
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

    size = len(range(int(start), int(stop), int(step)))
    if size == 0:
        return as_column([], dtype=dtype)

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


def concat_columns(objs: "MutableSequence[ColumnBase]") -> ColumnBase:
    """Concatenate a sequence of columns."""
    if len(objs) == 0:
        dtype = cudf.dtype(None)
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
            if "exceeds size_type range" in str(e):
                raise OverflowError(
                    "total size of output is too large for a cudf column"
                ) from e
            raise
    return col
