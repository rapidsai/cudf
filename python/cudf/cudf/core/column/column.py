# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import builtins
import pickle
from collections import abc
from functools import cached_property
from itertools import chain
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from numba import cuda
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from typing_extensions import Self

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
from cudf._lib.types import size_type_dtype
from cudf._typing import ColumnLike, Dtype, ScalarLike
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    _is_pandas_nullable_extension_dtype,
    infer_dtype,
    is_bool_dtype,
    is_dtype_equal,
    is_scalar,
    is_string_dtype,
)
from cudf.core._compat import PANDAS_GE_210
from cudf.core.abc import Serializable
from cudf.core.buffer import (
    Buffer,
    acquire_spill_lock,
    as_buffer,
    cuda_array_interface_wrapper,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    DecimalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.mixins import BinaryOperand, Reducible
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    _maybe_convert_to_default_type,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    find_common_type,
    get_time_unit,
    is_column_like,
    is_mixed_with_object_dtype,
    min_scalar_type,
    min_unsigned_type,
)
from cudf.utils.utils import _array_ufunc, mask_dtype

if PANDAS_GE_210:
    NumpyExtensionArray = pd.arrays.NumpyExtensionArray
else:
    NumpyExtensionArray = pd.arrays.PandasArray


class ColumnBase(Column, Serializable, BinaryOperand, Reducible):
    _VALID_REDUCTIONS = {
        "any",
        "all",
        "max",
        "min",
    }

    def data_array_view(
        self, *, mode: Literal["write", "read"] = "write"
    ) -> "cuda.devicearray.DeviceNDArray":
        """
        View the data as a device array object

        Parameters
        ----------
        mode : str, default 'write'
            Supported values are {'read', 'write'}
            If 'write' is passed, a device array object
            with readonly flag set to False in CAI is returned.
            If 'read' is passed, a device array object
            with readonly flag set to True in CAI is returned.
            This also means, If the caller wishes to modify
            the data returned through this view, they must
            pass mode="write", else pass mode="read".

        Returns
        -------
        numba.cuda.cudadrv.devicearray.DeviceNDArray
        """
        if self.data is not None:
            if mode == "read":
                obj = cuda_array_interface_wrapper(
                    ptr=self.data.get_ptr(mode="read"),
                    size=self.data.size,
                    owner=self.data,
                )
            elif mode == "write":
                obj = self.data
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        else:
            obj = None
        return cuda.as_cuda_array(obj).view(self.dtype)

    def mask_array_view(
        self, *, mode: Literal["write", "read"] = "write"
    ) -> "cuda.devicearray.DeviceNDArray":
        """
        View the mask as a device array

        Parameters
        ----------
        mode : str, default 'write'
            Supported values are {'read', 'write'}
            If 'write' is passed, a device array object
            with readonly flag set to False in CAI is returned.
            If 'read' is passed, a device array object
            with readonly flag set to True in CAI is returned.
            This also means, If the caller wishes to modify
            the data returned through this view, they must
            pass mode="write", else pass mode="read".

        Returns
        -------
        numba.cuda.cudadrv.devicearray.DeviceNDArray
        """
        if self.mask is not None:
            if mode == "read":
                obj = cuda_array_interface_wrapper(
                    ptr=self.mask.get_ptr(mode="read"),
                    size=self.mask.size,
                    owner=self.mask,
                )
            elif mode == "write":
                obj = self.mask
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        else:
            obj = None
        return cuda.as_cuda_array(obj).view(mask_dtype)

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        return (
            f"{object.__repr__(self)}\n"
            f"{self.to_arrow().to_string()}\n"
            f"dtype: {self.dtype}"
        )

    def to_pandas(
        self,
        *,
        index: Optional[pd.Index] = None,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Series:
        """Convert object to pandas type.

        The default implementation falls back to PyArrow for the conversion.
        """
        # This default implementation does not handle nulls in any meaningful
        # way
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        elif nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        pa_array = self.to_arrow()
        if arrow_type:
            return pd.Series(
                pd.arrays.ArrowExtensionArray(pa_array), index=index
            )
        else:
            pd_series = pa_array.to_pandas()

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

        with acquire_spill_lock():
            return self.data_array_view(mode="read").copy_to_host()

    @property
    def values(self) -> "cupy.ndarray":
        """
        Return a CuPy representation of the Column.
        """
        if len(self) == 0:
            return cupy.array([], dtype=self.dtype)

        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        return cupy.asarray(self.data_array_view(mode="write"))

    def find_and_replace(
        self,
        to_replace: ColumnLike,
        replacement: ColumnLike,
        all_nan: bool = False,
    ) -> Self:
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

    def dropna(self) -> ColumnBase:
        return drop_nulls([self])[0]._with_type_metadata(self.dtype)

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
        return libcudf.interop.to_arrow([self], [("None", self.dtype)])[
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

        if (
            isinstance(array.type, pa.TimestampType)
            and array.type.tz is not None
        ):
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
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
        elif isinstance(array.type, ArrowIntervalType):
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
    ) -> Optional[Self]:
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
    def nullmask(self) -> Buffer:
        """The gpu buffer for the null-mask"""
        if not self.nullable:
            raise ValueError("Column has no null mask")
        return self.mask_array_view(mode="read")

    def copy(self, deep: bool = True) -> Self:
        """
        Makes a copy of the Column.

        Parameters
        ----------
        deep : bool, default True
            If True, a true physical copy of the column
            is made.
            If False and `copy_on_write` is False, the same
            memory is shared between the buffers of the Column
            and changes made to one Column will propagate to
            its copy and vice-versa.
            If False and `copy_on_write` is True, the same
            memory is shared between the buffers of the Column
            until there is a write operation being performed on
            them.
        """
        if deep:
            result = libcudf.copying.copy_column(self)
            return result._with_type_metadata(self.dtype)
        else:
            return cast(
                Self,
                build_column(
                    data=self.base_data
                    if self.base_data is None
                    else self.base_data.copy(deep=False),
                    dtype=self.dtype,
                    mask=self.base_mask
                    if self.base_mask is None
                    else self.base_mask.copy(deep=False),
                    size=self.size,
                    offset=self.offset,
                    children=tuple(
                        col.copy(deep=False) for col in self.base_children
                    ),
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

    def slice(
        self, start: int, stop: int, stride: Optional[int] = None
    ) -> Self:
        stride = 1 if stride is None else stride
        if start < 0:
            start = start + len(self)
        if stop < 0 and not (stride < 0 and stop == -1):
            stop = stop + len(self)
        if (stride > 0 and start >= stop) or (stride < 0 and start <= stop):
            return cast(Self, column_empty(0, self.dtype, masked=True))
        # compute mask slice
        if stride == 1:
            return libcudf.copying.column_slice(self, [start, stop])[
                0
            ]._with_type_metadata(self.dtype)
        else:
            # Need to create a gather map for given slice with stride
            gather_map = as_column(
                range(start, stop, stride),
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
        if cudf.utils.utils.is_na_like(other):
            return cudf.Scalar(other, dtype=self.dtype)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            # Try and maintain the dtype
            other = other.dtype.type(other.item())
        return self.normalize_binop_value(other)

    def _scatter_by_slice(
        self,
        key: builtins.slice,
        value: Union[cudf.core.scalar.Scalar, ColumnBase],
    ) -> Optional[Self]:
        """If this function returns None, it's either a no-op (slice is empty),
        or the inplace replacement is already performed (fill-in-place).
        """
        start, stop, step = key.indices(len(self))
        if start >= stop:
            return None
        rng = range(start, stop, step)
        num_keys = len(rng)

        self._check_scatter_key_length(num_keys, value)

        if step == 1 and not isinstance(
            self, (cudf.core.column.StructColumn, cudf.core.column.ListColumn)
        ):
            # NOTE: List & Struct dtypes aren't supported by both
            # inplace & out-of-place fill. Hence we need to use scatter for
            # these two types.
            if isinstance(value, cudf.core.scalar.Scalar):
                return self._fill(value, start, stop, inplace=True)
            else:
                return libcudf.copying.copy_range(
                    value, self, 0, num_keys, start, stop, False
                )

        # step != 1, create a scatter map with arange
        scatter_map = as_column(
            rng,
            dtype=cudf.dtype(np.int32),
        )

        return self._scatter_by_column(scatter_map, value)

    def _scatter_by_column(
        self,
        key: cudf.core.column.NumericalColumn,
        value: Union[cudf.core.scalar.Scalar, ColumnBase],
    ) -> Self:
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

        if is_bool_dtype(key.dtype):
            return libcudf.copying.boolean_mask_scatter([value], [self], key)[
                0
            ]._with_type_metadata(self.dtype)
        else:
            return libcudf.copying.scatter([value], key, [self])[
                0
            ]._with_type_metadata(self.dtype)

    def _check_scatter_key_length(
        self, num_keys: int, value: Union[cudf.core.scalar.Scalar, ColumnBase]
    ) -> None:
        """`num_keys` is the number of keys to scatter. Should equal to the
        number of rows in ``value`` if ``value`` is a column.
        """
        if isinstance(value, ColumnBase) and len(value) != num_keys:
            raise ValueError(
                f"Size mismatch: cannot set value "
                f"of size {len(value)} to indexing result of size "
                f"{num_keys}"
            )

    def fillna(
        self,
        fill_value: Any = None,
        method: Optional[str] = None,
    ) -> Self:
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        return libcudf.replace.replace_nulls(
            input_col=self, replacement=fill_value, method=method
        )._with_type_metadata(self.dtype)

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column."""
        result = libcudf.unary.is_null(self)

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of a float column
            result = result | libcudf.unary.is_nan(self)

        return result

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column."""
        result = libcudf.unary.is_valid(self)

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of a float column
            result = result & libcudf.unary.is_non_nan(self)

        return result

    def indices_of(
        self, value: ScalarLike | Self
    ) -> cudf.core.column.NumericalColumn:
        """
        Find locations of value in the column

        Parameters
        ----------
        value
            Scalar to look for (cast to dtype of column), or a length-1 column

        Returns
        -------
        Column of indices that match value
        """
        if not isinstance(value, ColumnBase):
            value = as_column([value], dtype=self.dtype)
        else:
            assert len(value) == 1
        mask = libcudf.search.contains(value, self)
        return apply_boolean_mask(
            [as_column(range(0, len(self)), dtype=size_type_dtype)], mask
        )[0]

    def _find_first_and_last(self, value: ScalarLike) -> Tuple[int, int]:
        indices = self.indices_of(value)
        if n := len(indices):
            return (
                indices.element_indexing(0),
                indices.element_indexing(n - 1),
            )
        else:
            raise ValueError(f"Value {value} not found in column")

    def find_first_value(self, value: ScalarLike) -> int:
        """
        Return index of first value that matches

        Parameters
        ----------
        value
            Value to search for (cast to dtype of column)

        Returns
        -------
        Index of value

        Raises
        ------
        ValueError if value is not found
        """
        first, _ = self._find_first_and_last(value)
        return first

    def find_last_value(self, value: ScalarLike) -> int:
        """
        Return index of last value that matches

        Parameters
        ----------
        value
            Value to search for (cast to dtype of column)

        Returns
        -------
        Index of value

        Raises
        ------
        ValueError if value is not found
        """
        _, last = self._find_first_and_last(value)
        return last

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
        self, indices: ColumnBase, nullify: bool = False, check_bounds=True
    ) -> Self:
        """Return Column by taking values from the corresponding *indices*.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        # Handle zero size
        if indices.size == 0:
            return cast(Self, column_empty_like(self, newsize=0))

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if indices.dtype.kind not in {"u", "i"}:
            indices = indices.astype(libcudf.types.size_type_dtype)
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
            return as_column(False, length=len(self), dtype="bool")

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
                return as_column(False, length=len(self), dtype="bool")
        elif self.null_count == 0 and (rhs.null_count == len(rhs)):
            return as_column(False, length=len(self), dtype="bool")
        else:
            return None

    def _obtain_isin_result(self, rhs: ColumnBase) -> ColumnBase:
        """
        Helper function for `isin` which merges `self` & `rhs`
        to determine what values of `rhs` exist in `self`.
        """
        # We've already matched dtypes by now
        # self.isin(other) asks "which values of self are in other"
        # contains(haystack, needles) asks "which needles are in haystack"
        # hence this argument ordering.
        result = libcudf.search.contains(rhs, self)
        if self.null_count > 0:
            # If one of the needles is null, then the result contains
            # nulls, these nulls should be replaced by whether or not the
            # haystack contains a null.
            # TODO: this is unnecessary if we resolve
            # https://github.com/rapidsai/cudf/issues/14515 by
            # providing a mode in which cudf::contains does not mask
            # the result.
            result = result.fillna(cudf.Scalar(rhs.null_count > 0))
        return result

    def as_mask(self) -> Buffer:
        """Convert booleans to bitmask

        Returns
        -------
        Buffer
        """

        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        return bools_to_mask(self)

    @property
    def is_unique(self) -> bool:
        return self.distinct_count(dropna=False) == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        return not self.has_nulls() and libcudf.sort.is_sorted(
            [self], [True], None
        )

    @property
    def is_monotonic_decreasing(self) -> bool:
        return not self.has_nulls() and libcudf.sort.is_sorted(
            [self], [False], None
        )

    def sort_values(
        self: ColumnBase,
        ascending: bool = True,
        na_position: str = "last",
    ) -> ColumnBase:
        return libcudf.sort.sort(
            [self], column_order=[ascending], null_precedence=[na_position]
        )[0]

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

    def astype(self, dtype: Dtype, copy: bool = False) -> ColumnBase:
        if copy:
            col = self.copy()
        else:
            col = self
        if dtype == "category":
            # TODO: Figure out why `cudf.dtype("category")`
            # astype's different than just the string
            return col.as_categorical_column(dtype)
        elif (
            isinstance(dtype, str)
            and dtype == "interval"
            and isinstance(self.dtype, cudf.IntervalDtype)
        ):
            # astype("interval") (the string only) should no-op
            return col
        was_object = dtype == object or dtype == np.dtype(object)
        dtype = cudf.dtype(dtype)
        if self.dtype == dtype:
            return col
        elif isinstance(dtype, CategoricalDtype):
            return col.as_categorical_column(dtype)
        elif isinstance(dtype, IntervalDtype):
            return col.as_interval_column(dtype)
        elif isinstance(dtype, (ListDtype, StructDtype)):
            if not col.dtype == dtype:
                raise NotImplementedError(
                    f"Casting {self.dtype} columns not currently supported"
                )
            return col
        elif isinstance(dtype, cudf.core.dtypes.DecimalDtype):
            return col.as_decimal_column(dtype)
        elif np.issubdtype(cast(Any, dtype), np.datetime64):
            return col.as_datetime_column(dtype)
        elif np.issubdtype(cast(Any, dtype), np.timedelta64):
            return col.as_timedelta_column(dtype)
        elif dtype.kind == "O":
            if cudf.get_option("mode.pandas_compatible") and was_object:
                raise ValueError(
                    f"Casting to {dtype} is not supported, use "
                    "`.astype('str')` instead."
                )
            return col.as_string_column(dtype)
        else:
            return col.as_numerical_column(dtype)

    def as_categorical_column(self, dtype) -> ColumnBase:
        if isinstance(dtype, (cudf.CategoricalDtype, pd.CategoricalDtype)):
            ordered = dtype.ordered
        else:
            ordered = False

        # Re-label self w.r.t. the provided categories
        if (
            isinstance(dtype, cudf.CategoricalDtype)
            and dtype._categories is not None
        ) or (
            isinstance(dtype, pd.CategoricalDtype)
            and dtype.categories is not None
        ):
            labels = self._label_encoding(cats=as_column(dtype.categories))

            return build_categorical_column(
                categories=as_column(dtype.categories),
                codes=labels,
                mask=self.mask,
                ordered=dtype.ordered,
            )

        # Categories must be unique and sorted in ascending order.
        cats = self.unique().sort_values().astype(self.dtype)
        label_dtype = min_unsigned_type(len(cats))
        labels = self._label_encoding(
            cats=cats, dtype=label_dtype, na_sentinel=cudf.Scalar(1)
        )
        # columns include null index in factorization; remove:
        if self.has_nulls():
            cats = cats.dropna()
            min_type = min_unsigned_type(len(cats), 8)
            if cudf.dtype(min_type).itemsize < labels.dtype.itemsize:
                labels = labels.astype(min_type)

        return build_categorical_column(
            categories=cats,
            codes=labels,
            mask=self.mask,
            ordered=ordered,
        )

    def as_numerical_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.NumericalColumn":
        raise NotImplementedError

    def as_datetime_column(
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.DatetimeColumn":
        raise NotImplementedError

    def as_interval_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.IntervalColumn":
        raise NotImplementedError

    def as_timedelta_column(
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.TimeDeltaColumn":
        raise NotImplementedError

    def as_string_column(
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.StringColumn":
        raise NotImplementedError

    def as_decimal_column(
        self, dtype: Dtype
    ) -> Union["cudf.core.column.decimal.DecimalBaseColumn"]:
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
    ) -> "cudf.core.column.NumericalColumn":
        return libcudf.sort.order_by(
            [self], [ascending], na_position, stable=True
        )

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    @property
    def __cuda_array_interface__(self) -> abc.Mapping[str, Any]:
        output = {
            "shape": (len(self),),
            "strides": (self.dtype.itemsize,),
            "typestr": self.dtype.str,
            "data": (self.data_ptr, False),
            "version": 1,
        }

        if self.nullable and self.has_nulls():
            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            output["mask"] = cuda_array_interface_wrapper(
                ptr=self.mask_ptr,
                size=len(self),
                owner=self.mask,
                readonly=True,
                typestr="<t1",
            )
        return output

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    def searchsorted(
        self,
        value,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> Self:
        if not isinstance(value, ColumnBase) or value.dtype != self.dtype:
            raise ValueError(
                "Column searchsorted expects values to be column of same dtype"
            )
        return libcudf.search.search_sorted(
            [self],
            [value],
            side=side,
            ascending=ascending,
            na_position=na_position,
        )

    def unique(self) -> ColumnBase:
        """
        Get unique values in the data
        """
        return drop_duplicates([self], keep="first")[0]._with_type_metadata(
            self.dtype
        )

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

    def _reduce(
        self,
        op: str,
        skipna: Optional[bool] = None,
        min_count: int = 0,
        *args,
        **kwargs,
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

    def _process_for_reduction(
        self, skipna: Optional[bool] = None, min_count: int = 0
    ) -> Union[ColumnBase, ScalarLike]:
        if skipna is None:
            skipna = True

        if self.has_nulls():
            if skipna:
                result_col = self.dropna()
            else:
                return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        result_col = self

        # TODO: If and when pandas decides to validate that `min_count` >= 0 we
        # should insert comparable behavior.
        # https://github.com/pandas-dev/pandas/issues/50022
        if min_count > 0:
            valid_count = len(result_col) - result_col.null_count
            if valid_count < min_count:
                return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
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

    def _label_encoding(
        self,
        cats: ColumnBase,
        dtype: Optional[Dtype] = None,
        na_sentinel: Optional[ScalarLike] = None,
    ):
        """
        Convert each value in `self` into an integer code, with `cats`
        providing the mapping between codes and values.

        Examples
        --------
        >>> from cudf.core.column import as_column
        >>> col = as_column(['foo', 'bar', 'foo', 'baz'])
        >>> cats = as_column(['foo', 'bar', 'baz'])
        >>> col._label_encoding(cats)
        <cudf.core.column.numerical.NumericalColumn object at 0x7f99bf3155c0>
        [
          0,
          1,
          0,
          2
        ]
        dtype: int8
        >>> cats = as_column(['foo', 'bar'])
        >>> col._label_encoding(cats)
        <cudf.core.column.numerical.NumericalColumn object at 0x7f99bfde0e40>
        [
          0,
          1,
          0,
          -1
        ]
        dtype: int8
        """
        from cudf._lib.join import join as cpp_join

        if na_sentinel is None or na_sentinel.value is cudf.NA:
            na_sentinel = cudf.Scalar(-1)

        def _return_sentinel_column():
            return as_column(na_sentinel, dtype=dtype, length=len(self))

        if dtype is None:
            dtype = min_scalar_type(max(len(cats), na_sentinel), 8)

        if is_mixed_with_object_dtype(self, cats):
            return _return_sentinel_column()

        try:
            # Where there is a type-cast failure, we have
            # to catch the exception and return encoded labels
            # with na_sentinel values as there would be no corresponding
            # encoded values of cats in self.
            cats = cats.astype(self.dtype)
        except ValueError:
            return _return_sentinel_column()

        left_gather_map, right_gather_map = cpp_join(
            [self], [cats], how="left"
        )
        codes = libcudf.copying.gather(
            [as_column(range(len(cats)), dtype=dtype)],
            right_gather_map,
            nullify=True,
        )
        del right_gather_map
        # reorder `codes` so that its values correspond to the
        # values of `self`:
        (codes,) = libcudf.sort.sort_by_key(
            codes, [left_gather_map], [True], ["last"], stable=True
        )
        return codes.fillna(na_sentinel.value)


def column_empty_like(
    column: ColumnBase,
    dtype: Optional[Dtype] = None,
    masked: bool = False,
    newsize: Optional[int] = None,
) -> ColumnBase:
    """Allocate a new column like the given *column*"""
    if dtype is None:
        dtype = column.dtype
    row_count = len(column) if newsize is None else newsize

    if (
        hasattr(column, "dtype")
        and isinstance(column.dtype, cudf.CategoricalDtype)
        and dtype == column.dtype
    ):
        catcolumn = cast("cudf.core.column.CategoricalColumn", column)
        codes = column_empty_like(
            catcolumn.codes, masked=masked, newsize=newsize
        )
        return build_column(
            data=None,
            dtype=dtype,
            mask=codes.base_mask,
            children=(codes,),
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

    if isinstance(dtype, StructDtype):
        data = None
        children = tuple(
            column_empty(row_count, field_dtype)
            for field_dtype in dtype.fields.values()
        )
    elif isinstance(dtype, ListDtype):
        data = None
        children = (
            as_column(
                0, length=row_count + 1, dtype=libcudf.types.size_type_dtype
            ),
            column_empty(row_count, dtype=dtype.element_type),
        )
    elif isinstance(dtype, CategoricalDtype):
        data = None
        children = (
            build_column(
                data=as_buffer(
                    rmm.DeviceBuffer(
                        size=row_count
                        * cudf.dtype(libcudf.types.size_type_dtype).itemsize
                    )
                ),
                dtype=libcudf.types.size_type_dtype,
            ),
        )
    elif dtype.kind in "OU" and not isinstance(dtype, DecimalDtype):
        data = as_buffer(rmm.DeviceBuffer(size=0))
        children = (
            as_column(
                0, length=row_count + 1, dtype=libcudf.types.size_type_dtype
            ),
        )
    else:
        data = as_buffer(rmm.DeviceBuffer(size=row_count * dtype.itemsize))

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
    size: Optional[int] = None,
    mask: Optional[Buffer] = None,
    offset: int = 0,
    null_count: Optional[int] = None,
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
    dtype = cudf.dtype(dtype)

    if _is_non_decimal_numeric_dtype(dtype):
        assert data is not None
        col = cudf.core.column.NumericalColumn(
            data=data,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
        )
        return col

    if isinstance(dtype, CategoricalDtype):
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
    elif isinstance(dtype, pd.DatetimeTZDtype):
        if data is None:
            raise TypeError("Must specify data buffer")
        return cudf.core.column.datetime.DatetimeTZColumn(
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
            data=data,
            mask=mask,
            size=size,
            offset=offset,
            children=children,
            null_count=null_count,
        )
    elif isinstance(dtype, ListDtype):
        return cudf.core.column.ListColumn(
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )
    elif isinstance(dtype, IntervalDtype):
        return cudf.core.column.IntervalColumn(
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            children=children,
            null_count=null_count,
        )
    elif isinstance(dtype, StructDtype):
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
    elif isinstance(dtype, cudf.Decimal64Dtype):
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
    elif isinstance(dtype, cudf.Decimal32Dtype):
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
    elif isinstance(dtype, cudf.Decimal128Dtype):
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
    else:
        raise TypeError(f"Unrecognized dtype: {dtype}")


def build_categorical_column(
    categories: ColumnBase,
    codes: ColumnBase,
    mask: Optional[Buffer] = None,
    size: Optional[int] = None,
    offset: int = 0,
    null_count: Optional[int] = None,
    ordered: bool = False,
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
    ordered : bool, default False
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


def check_invalid_array(shape: tuple, dtype):
    """Invalid ndarrays properties that are not supported"""
    if len(shape) > 1:
        raise ValueError("Data must be 1-dimensional")
    elif dtype == "float16":
        raise TypeError("Unsupported type float16")


def as_memoryview(arbitrary: Any) -> Optional[memoryview]:
    try:
        return memoryview(arbitrary)
    except TypeError:
        return None


def as_column(
    arbitrary: Any,
    nan_as_null: Optional[bool] = None,
    dtype: Optional[Dtype] = None,
    length: Optional[int] = None,
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
        Only applies when arbitrary is not a cudf object
        (Index, Series, Column).
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
    * range objects
    """
    if isinstance(arbitrary, (range, pd.RangeIndex, cudf.RangeIndex)):
        column = libcudf.filling.sequence(
            len(arbitrary),
            as_device_scalar(arbitrary.start, dtype=cudf.dtype("int64")),
            as_device_scalar(arbitrary.step, dtype=cudf.dtype("int64")),
        )
        if cudf.get_option("default_integer_bitwidth") and dtype is None:
            dtype = cudf.dtype(
                f'i{cudf.get_option("default_integer_bitwidth")//8}'
            )
        if dtype is not None:
            return column.astype(dtype)
        return column
    elif isinstance(arbitrary, (ColumnBase, cudf.Series, cudf.BaseIndex)):
        # Ignoring nan_as_null per the docstring
        if isinstance(arbitrary, cudf.Series):
            arbitrary = arbitrary._column
        elif isinstance(arbitrary, cudf.BaseIndex):
            arbitrary = arbitrary._values
        if dtype is not None:
            return arbitrary.astype(dtype)
        return arbitrary
    elif hasattr(arbitrary, "__cuda_array_interface__"):
        desc = arbitrary.__cuda_array_interface__
        check_invalid_array(desc["shape"], np.dtype(desc["typestr"]))

        if desc.get("mask", None) is not None:
            # Extract and remove the mask from arbitrary before
            # passing to cupy.asarray
            cai_copy = desc.copy()
            mask = _mask_from_cuda_array_interface_desc(
                arbitrary, cai_copy.pop("mask")
            )
            arbitrary = SimpleNamespace(__cuda_array_interface__=cai_copy)
        else:
            mask = None

        arbitrary = cupy.asarray(arbitrary)
        arbitrary = cupy.ascontiguousarray(arbitrary)

        data = as_buffer(arbitrary, exposed=cudf.get_option("copy_on_write"))
        col = build_column(data, dtype=arbitrary.dtype, mask=mask)
        if (
            nan_as_null or (mask is None and nan_as_null is None)
        ) and col.dtype.kind == "f":
            col = col.nans_to_nulls()
        if dtype is not None:
            col = col.astype(dtype)
        return col

    elif isinstance(arbitrary, (pa.Array, pa.ChunkedArray)):
        if pa.types.is_large_string(arbitrary.type):
            # Pandas-2.2+: Pandas defaults to `large_string` type
            # instead of `string` without data-introspection.
            # Temporary workaround until cudf has native
            # support for `LARGE_STRING` i.e., 64 bit offsets
            arbitrary = arbitrary.cast(pa.string())

        if pa.types.is_float16(arbitrary.type):
            raise NotImplementedError(
                "Type casting from `float16` to `float32` is not "
                "yet supported in pyarrow, see: "
                "https://github.com/apache/arrow/issues/20213"
            )
        elif (
            pa.types.is_timestamp(arbitrary.type)
            and arbitrary.type.tz is not None
        ):
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        elif (nan_as_null is None or nan_as_null) and pa.types.is_floating(
            arbitrary.type
        ):
            arbitrary = pc.if_else(
                pc.is_nan(arbitrary),
                pa.nulls(len(arbitrary), type=arbitrary.type),
                arbitrary,
            )
        col = ColumnBase.from_arrow(arbitrary)

        if isinstance(arbitrary, pa.NullArray):
            if dtype is not None:
                # Cast the column to the `dtype` if specified.
                new_dtype = dtype
            elif len(arbitrary) == 0:
                # If the column is empty, it has to be
                # a `str` dtype.
                new_dtype = cudf.dtype("str")
            else:
                # If the null column is not empty, it has to
                # be of `object` dtype.
                new_dtype = cudf.dtype(arbitrary.type.to_pandas_dtype())

            if cudf.get_option(
                "mode.pandas_compatible"
            ) and new_dtype == cudf.dtype("O"):
                # We internally raise if we do `astype("object")`, hence
                # need to cast to `str` since this is safe to do so because
                # it is a null-array.
                new_dtype = "str"

            col = col.astype(new_dtype)
        elif dtype is not None:
            col = col.astype(dtype)

        return col

    elif isinstance(
        arbitrary, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)
    ):
        if isinstance(arbitrary.dtype, (pd.SparseDtype, pd.PeriodDtype)):
            raise NotImplementedError(
                f"cuDF does not yet support {type(arbitrary.dtype).__name__}"
            )
        elif (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(arbitrary, (pd.DatetimeIndex, pd.TimedeltaIndex))
            and arbitrary.freq is not None
        ):
            raise NotImplementedError("freq is not implemented yet")
        elif (
            isinstance(arbitrary.dtype, pd.DatetimeTZDtype)
            or (
                isinstance(arbitrary.dtype, pd.IntervalDtype)
                and isinstance(arbitrary.dtype.subtype, pd.DatetimeTZDtype)
            )
            or (
                isinstance(arbitrary.dtype, pd.CategoricalDtype)
                and isinstance(
                    arbitrary.dtype.categories.dtype, pd.DatetimeTZDtype
                )
            )
        ):
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        elif _is_pandas_nullable_extension_dtype(arbitrary.dtype):
            if cudf.get_option("mode.pandas_compatible"):
                raise NotImplementedError("not supported")
            if isinstance(arbitrary, (pd.Series, pd.Index)):
                # pandas arrays define __arrow_array__ for better
                # pyarrow.array conversion
                arbitrary = arbitrary.array
            return as_column(
                pa.array(arbitrary, from_pandas=True),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
        elif isinstance(
            arbitrary.dtype, (pd.CategoricalDtype, pd.IntervalDtype)
        ):
            return as_column(
                pa.array(arbitrary, from_pandas=True),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
        elif isinstance(
            arbitrary.dtype, pd.api.extensions.ExtensionDtype
        ) and not isinstance(arbitrary, NumpyExtensionArray):
            raise NotImplementedError(
                "Custom pandas ExtensionDtypes are not supported"
            )
        elif arbitrary.dtype.kind in "fiubmM":
            # numpy dtype like
            if isinstance(arbitrary, NumpyExtensionArray):
                arbitrary = np.array(arbitrary)
            arb_dtype = np.dtype(arbitrary.dtype)
            if arb_dtype.kind == "f" and arb_dtype.itemsize == 2:
                raise TypeError("Unsupported type float16")
            elif arb_dtype.kind in "mM":
                # not supported by cupy
                arbitrary = np.asarray(arbitrary)
            else:
                arbitrary = cupy.asarray(arbitrary)
            return as_column(
                arbitrary, nan_as_null=nan_as_null, dtype=dtype, length=length
            )
        elif arbitrary.dtype.kind == "O":
            if isinstance(arbitrary, NumpyExtensionArray):
                # infer_dtype does not handle NumpyExtensionArray
                arbitrary = np.array(arbitrary, dtype=object)
            inferred_dtype = infer_dtype(arbitrary)
            if inferred_dtype in ("mixed-integer", "mixed-integer-float"):
                raise MixedTypeError("Cannot create column with mixed types")
            elif dtype is None and inferred_dtype not in (
                "mixed",
                "decimal",
                "string",
                "empty",
                "boolean",
            ):
                raise TypeError(
                    f"Cannot convert a {inferred_dtype} of object type"
                )
            elif nan_as_null is False and (
                pd.isna(arbitrary).any()
                and inferred_dtype not in ("decimal", "empty")
            ):
                # Decimal can hold float("nan")
                # All np.nan is not restricted by type
                raise MixedTypeError(f"Cannot have NaN with {inferred_dtype}")

            pyarrow_array = pa.array(
                arbitrary,
                from_pandas=True,
            )
            return as_column(
                pyarrow_array,
                dtype=dtype,
                nan_as_null=nan_as_null,
                length=length,
            )
        else:
            raise NotImplementedError(
                f"{type(arbitrary).__name__} with "
                f"{type(arbitrary.dtype).__name__} is not supported."
            )
    elif is_scalar(arbitrary) and not isinstance(arbitrary, memoryview):
        if length is None:
            length = 1
        elif length < 0:
            raise ValueError(f"{length=} must be >=0.")
        if isinstance(
            arbitrary, pd.Interval
        ) or cudf.api.types._is_categorical_dtype(dtype):
            # No cudf.Scalar support yet
            return as_column(
                pd.Series([arbitrary] * length),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
        if (
            nan_as_null is True
            and isinstance(arbitrary, (np.floating, float))
            and np.isnan(arbitrary)
        ):
            if dtype is None:
                dtype = getattr(arbitrary, "dtype", cudf.dtype("float64"))
            arbitrary = None
        arbitrary = cudf.Scalar(arbitrary, dtype=dtype)
        if length == 0:
            return column_empty(length, dtype=arbitrary.dtype)
        else:
            return ColumnBase.from_scalar(arbitrary, length)

    elif hasattr(arbitrary, "__array_interface__"):
        desc = arbitrary.__array_interface__
        check_invalid_array(desc["shape"], np.dtype(desc["typestr"]))

        # CUDF assumes values are always contiguous
        arbitrary = np.asarray(arbitrary, order="C")

        if arbitrary.ndim == 0:
            # TODO: Or treat as scalar?
            arbitrary = arbitrary[np.newaxis]

        if arbitrary.dtype.kind in "OSU":
            if pd.isna(arbitrary).any():
                arbitrary = pa.array(arbitrary)
            else:
                # Let pandas potentially infer object type
                # e.g. np.array([pd.Timestamp(...)], dtype=object) -> datetime64
                arbitrary = pd.Series(arbitrary)
            return as_column(arbitrary, dtype=dtype, nan_as_null=nan_as_null)
        elif arbitrary.dtype.kind in "biuf":
            from_pandas = nan_as_null is None or nan_as_null
            return as_column(
                pa.array(arbitrary, from_pandas=from_pandas),
                dtype=dtype,
                nan_as_null=nan_as_null,
            )
        elif arbitrary.dtype.kind in "mM":
            time_unit = get_time_unit(arbitrary)
            if time_unit in ("D", "W", "M", "Y"):
                # TODO: Raise in these cases instead of downcasting to s?
                new_type = f"{arbitrary.dtype.type.__name__}[s]"
                arbitrary = arbitrary.astype(new_type)
            elif time_unit == "generic":
                # TODO: This should probably be in cudf.dtype
                raise TypeError(
                    f"{arbitrary.dtype.type.__name__} must have a unit specified"
                )

            is_nat = np.isnat(arbitrary)
            mask = None
            if is_nat.any():
                if nan_as_null is None or nan_as_null:
                    # Convert NaT to NA, which pyarrow does by default
                    return as_column(
                        pa.array(arbitrary),
                        dtype=dtype,
                        nan_as_null=nan_as_null,
                    )
                # Consider NaT as NA in the mask
                # but maintain NaT as a value
                bool_mask = as_column(~is_nat)
                mask = as_buffer(bools_to_mask(bool_mask))
            buffer = as_buffer(arbitrary.view("|u1"))
            col = build_column(data=buffer, mask=mask, dtype=arbitrary.dtype)
            if dtype:
                col = col.astype(dtype)
            return col
        else:
            raise NotImplementedError(f"{arbitrary.dtype} not supported")
    elif (view := as_memoryview(arbitrary)) is not None:
        return as_column(
            np.asarray(view), dtype=dtype, nan_as_null=nan_as_null
        )
    elif hasattr(arbitrary, "__array__"):
        # e.g. test_cuda_array_interface_pytorch
        try:
            arbitrary = cupy.asarray(arbitrary)
        except (ValueError, TypeError):
            arbitrary = np.asarray(arbitrary)
        return as_column(arbitrary, dtype=dtype, nan_as_null=nan_as_null)
    # Start of arbitrary that's not handed above but dtype provided
    elif isinstance(dtype, pd.DatetimeTZDtype):
        raise NotImplementedError(
            "Use `tz_localize()` to construct timezone aware data."
        )
    elif isinstance(dtype, cudf.core.dtypes.DecimalDtype):
        # Arrow throws a type error if the input is of
        # mixed-precision and cannot fit into the provided
        # decimal type properly, see:
        # https://github.com/apache/arrow/pull/9948
        # Hence we should let the exception propagate to
        # the user.
        data = pa.array(
            arbitrary,
            type=pa.decimal128(precision=dtype.precision, scale=dtype.scale),
        )
        if isinstance(dtype, cudf.core.dtypes.Decimal128Dtype):
            return cudf.core.column.Decimal128Column.from_arrow(data)
        elif isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
            return cudf.core.column.Decimal64Column.from_arrow(data)
        elif isinstance(dtype, cudf.core.dtypes.Decimal32Dtype):
            return cudf.core.column.Decimal32Column.from_arrow(data)
        else:
            raise NotImplementedError(f"{dtype} not implemented")
    elif isinstance(
        dtype,
        (
            pd.CategoricalDtype,
            cudf.CategoricalDtype,
            pd.IntervalDtype,
            cudf.IntervalDtype,
        ),
    ) or dtype in {
        "category",
        "interval",
        "str",
        str,
        np.str_,
        object,
        np.dtype(object),
    }:
        if isinstance(dtype, (cudf.CategoricalDtype, cudf.IntervalDtype)):
            dtype = dtype.to_pandas()
        elif dtype == object:
            # Unlike pandas, interpret object as "str" instead of "python object"
            dtype = "str"
        ser = pd.Series(arbitrary, dtype=dtype)
        return as_column(ser, nan_as_null=nan_as_null)
    elif isinstance(dtype, (cudf.StructDtype, cudf.ListDtype)):
        try:
            data = pa.array(arbitrary, type=dtype.to_arrow())
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            if isinstance(dtype, cudf.ListDtype):
                # e.g. test_cudf_list_struct_write
                return cudf.core.column.ListColumn.from_sequences(arbitrary)
            raise
        return as_column(data, nan_as_null=nan_as_null)
    elif not isinstance(arbitrary, (abc.Iterable, abc.Sequence)):
        # TODO: This validation should probably be done earlier?
        raise TypeError(
            f"{type(arbitrary).__name__} must be an iterable or sequence."
        )
    from_pandas = nan_as_null is None or nan_as_null
    if dtype is not None:
        dtype = cudf.dtype(dtype)
        try:
            arbitrary = pa.array(
                arbitrary,
                type=cudf_dtype_to_pa_type(dtype),
                from_pandas=from_pandas,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            if not isinstance(dtype, np.dtype):
                dtype = dtype.to_pandas()
            arbitrary = pd.Series(arbitrary, dtype=dtype)
        return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)
    else:
        arbitrary = list(arbitrary)
        for element in arbitrary:
            # Carve-outs that cannot be parsed by pyarrow/pandas
            if is_column_like(element):
                # e.g. test_nested_series_from_sequence_data
                return cudf.core.column.ListColumn.from_sequences(arbitrary)
            elif isinstance(element, cupy.ndarray):
                # e.g. test_series_from_cupy_scalars
                return as_column(
                    cupy.array(arbitrary),
                    dtype=dtype,
                    nan_as_null=nan_as_null,
                    length=length,
                )
            elif (
                isinstance(element, (pd.Timestamp, pd.Timedelta))
                or element is pd.NaT
            ):
                # TODO: Remove this after
                # https://github.com/apache/arrow/issues/26492
                # is fixed.
                return as_column(
                    pd.Series(arbitrary),
                    dtype=dtype,
                    nan_as_null=nan_as_null,
                    length=length,
                )
            elif not any(element is na for na in (None, pd.NA, np.nan)):
                # Might have NA + element like above, but short-circuit if
                # an element pyarrow/pandas might be able to parse
                break
        try:
            arbitrary = pa.array(arbitrary, from_pandas=from_pandas)
            if (
                cudf.get_option("mode.pandas_compatible")
                and pa.types.is_integer(arbitrary.type)
                and arbitrary.null_count > 0
            ):
                arbitrary = arbitrary.cast(pa.float64())
            if cudf.get_option(
                "default_integer_bitwidth"
            ) and pa.types.is_integer(arbitrary.type):
                dtype = _maybe_convert_to_default_type("int")
            elif cudf.get_option(
                "default_float_bitwidth"
            ) and pa.types.is_floating(arbitrary.type):
                dtype = _maybe_convert_to_default_type("float")
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            arbitrary = pd.Series(arbitrary)
            if cudf.get_option(
                "default_integer_bitwidth"
            ) and arbitrary.dtype.kind in set("iu"):
                dtype = _maybe_convert_to_default_type("int")
            elif (
                cudf.get_option("default_float_bitwidth")
                and arbitrary.dtype.kind == "f"
            ):
                dtype = _maybe_convert_to_default_type("float")
        return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)


def _mask_from_cuda_array_interface_desc(obj, cai_mask) -> Buffer:
    desc = cai_mask.__cuda_array_interface__
    typestr = desc["typestr"]
    typecode = typestr[1]
    if typecode == "t":
        mask_size = bitmask_allocation_size_bytes(desc["shape"][0])
        return as_buffer(data=desc["data"][0], size=mask_size, owner=obj)
    elif typecode == "b":
        col = as_column(cai_mask)
        return bools_to_mask(col)
    else:
        raise NotImplementedError(f"Cannot infer mask from typestr {typestr}")


def serialize_columns(columns: list[ColumnBase]) -> Tuple[List[dict], List]:
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


def concat_columns(objs: "MutableSequence[ColumnBase]") -> ColumnBase:
    """Concatenate a sequence of columns."""
    if len(objs) == 0:
        dtype = cudf.dtype(None)
        return column_empty(0, dtype=dtype, masked=True)

    # If all columns are `NumericalColumn` with different dtypes,
    # we cast them to a common dtype.
    # Notice, we can always cast pure null columns
    not_null_col_dtypes = [o.dtype for o in objs if o.null_count != len(o)]
    if len(not_null_col_dtypes) and all(
        _is_non_decimal_numeric_dtype(dtyp)
        and np.issubdtype(dtyp, np.datetime64)
        for dtyp in not_null_col_dtypes
    ):
        common_dtype = find_common_type(not_null_col_dtypes)
        # Cast all columns to the common dtype
        objs = [obj.astype(common_dtype) for obj in objs]

    # Find the first non-null column:
    head = next((obj for obj in objs if obj.null_count != len(obj)), objs[0])

    for i, obj in enumerate(objs):
        # Check that all columns are the same type:
        if not is_dtype_equal(obj.dtype, head.dtype):
            # if all null, cast to appropriate dtype
            if obj.null_count == len(obj):
                objs[i] = column_empty_like(
                    head, dtype=head.dtype, masked=True, newsize=len(obj)
                )
            else:
                raise ValueError("All columns must be the same type")

    # TODO: This logic should be generalized to a dispatch to
    # ColumnBase._concat so that all subclasses can override necessary
    # behavior. However, at the moment it's not clear what that API should look
    # like, so CategoricalColumn simply implements a minimal working API.
    if all(isinstance(o.dtype, CategoricalDtype) for o in objs):
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
        return column_empty(0, head.dtype, masked=True)

    # Filter out inputs that have 0 length, then concatenate.
    return libcudf.concat.concat_columns([o for o in objs if len(o)])
