# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterable, Iterator, MutableSequence, Sequence
from functools import cached_property
from itertools import chain
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from packaging import version
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from typing_extensions import Self

import pylibcudf as plc

import cudf
from cudf.api.types import (
    _is_categorical_dtype,
    infer_dtype,
    is_dtype_equal,
    is_scalar,
)
from cudf.core._compat import PANDAS_GE_210
from cudf.core._internals import (
    aggregation,
    copying,
    sorting,
    stream_compaction,
)
from cudf.core._internals.timezones import get_compatible_timezone
from cudf.core.abc import Serializable
from cudf.core.buffer import (
    Buffer,
    acquire_spill_lock,
    as_buffer,
    cuda_array_interface_wrapper,
)
from cudf.core.buffer.spillable_buffer import SpillableBuffer
from cudf.core.copy_types import GatherMap
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
    CUDF_STRING_DTYPE,
    SIZE_TYPE_DTYPE,
    _get_nan_for_dtype,
    _maybe_convert_to_default_type,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    find_common_type,
    is_column_like,
    is_dtype_obj_decimal,
    is_dtype_obj_interval,
    is_dtype_obj_list,
    is_dtype_obj_numeric,
    is_dtype_obj_struct,
    is_mixed_with_object_dtype,
    is_pandas_nullable_extension_dtype,
    min_signed_type,
    np_dtypes_to_pandas_dtypes,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import (
    _array_ufunc,
    _is_null_host_scalar,
    is_na_like,
)

if TYPE_CHECKING:
    import builtins
    from collections.abc import Generator, Mapping

    from cudf._typing import ColumnLike, Dtype, DtypeObj, ScalarLike
    from cudf.core.column.categorical import CategoricalColumn
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.decimal import DecimalBaseColumn
    from cudf.core.column.interval import IntervalColumn
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.strings import StringColumn
    from cudf.core.column.timedelta import TimeDeltaColumn
    from cudf.core.index import Index

if PANDAS_GE_210:
    NumpyExtensionArray = pd.arrays.NumpyExtensionArray
else:
    NumpyExtensionArray = pd.arrays.PandasArray


def _can_values_be_equal(left: DtypeObj, right: DtypeObj) -> bool:
    """
    Given 2 possibly not equal dtypes, can they both hold equivalent values.

    Helper function for .equals when check_dtypes is False.
    """
    if left == right:
        return True
    if isinstance(left, CategoricalDtype):
        return _can_values_be_equal(left.categories.dtype, right)
    elif isinstance(right, CategoricalDtype):
        return _can_values_be_equal(left, right.categories.dtype)
    elif is_dtype_obj_numeric(left) and is_dtype_obj_numeric(right):
        return True
    elif left.kind == right.kind and left.kind in "mM":
        return True
    return False


class spillable_gpumemoryview(plc.gpumemoryview):
    """
    HACK: Prevent automatic unspilling of `SpillableBuffer` objects
    when constructing `plc.Column`.

    The `plc.Column()` constructor expects a `gpumemoryview` object,
    but wrapping a `SpillableBuffer` directly in a `gpumemoryview`
    forces the buffer to unspill (materialize) its device data prematurely.

    To avoid this, we wrap spillable buffers in this subclass that implements
    only the `.obj` attribute; the only attribute actually accessed by
    `.from_pylibcudf()`. All other attributes intentionally raise errors to
    prevent accidental usage paths that would cause unspilling.
    """

    def __init__(self, buf: SpillableBuffer) -> None:
        self._buf = buf

    @property
    def obj(self) -> SpillableBuffer:
        return self._buf

    @property
    def cai(self) -> None:  # type: ignore[override]
        assert False

    @property
    def ptr(self) -> None:  # type: ignore[override]
        assert False

    @property
    def nbytes(self) -> None:  # type: ignore[override]
        assert False


class ColumnBase(Serializable, BinaryOperand, Reducible):
    """
    A ColumnBase stores columnar data in device memory.

    A ColumnBase may be composed of:

    * A *data* Buffer
    * One or more (optional) *children* Columns
    * An (optional) *mask* Buffer representing the nullmask

    The *dtype* indicates the ColumnBase's element type.
    """

    _VALID_REDUCTIONS = {
        "any",
        "all",
        "max",
        "min",
    }
    _VALID_PLC_TYPES: ClassVar[set[plc.TypeId]] = set()

    def __init__(
        self,
        plc_column: plc.Column,
        size: int,
        dtype: DtypeObj,
        offset: int,
        null_count: int,
        exposed: bool,
    ) -> None:
        if not (
            isinstance(plc_column, plc.Column)
            and plc_column.type().id() in self._VALID_PLC_TYPES
        ):
            raise ValueError(
                f"plc_column must be a pylibcudf.Column with a TypeId in {self._VALID_PLC_TYPES}"
            )
        self.plc_column = plc_column
        if size < 0:
            raise ValueError("size must be >=0")
        self._size = size
        self._distinct_count: dict[bool, int] = {}
        self._dtype = dtype
        self._offset = offset
        if null_count < 0:
            raise ValueError("null_count must be >=0")
        self._null_count = null_count
        self._mask = None
        self._base_mask = None
        self._data = None
        self._children = None
        # CategoricalColumn overrides this method
        data = self._get_data_buffer_from_pylibcudf_column(
            self.plc_column, exposed
        )
        mask_view = plc_column.null_mask()
        mask = (
            as_buffer(mask_view.obj, exposed=exposed)
            if mask_view is not None
            else None
        )
        children = self._get_children_from_pylibcudf_column(
            self.plc_column,
            dtype,
            exposed,
        )
        self.set_base_children(children)
        self.set_base_data(data)
        self.set_base_mask(mask)

    @classmethod
    def _get_data_buffer_from_pylibcudf_column(
        cls, plc_column: plc.Column, exposed: bool
    ) -> Buffer | None:
        """
        Extract the data buffer from a pylibcudf.Column.

        Necessary to wrap the data buffer in a cuDF Buffer for spilling support.

        Parameters
        ----------
        plc_column : plc.Column
            The pylibcudf.Column to extract the data buffer from.
        exposed : bool
            Whether the data buffer is exposed.

        Returns
        -------
        Buffer | None
            The data buffer.
        """
        data_view = plc_column.data()
        return (
            as_buffer(data_view.obj, exposed=exposed)
            if data_view is not None
            else None
        )

    def _get_children_from_pylibcudf_column(
        self,
        plc_column: plc.Column,
        dtype: DtypeObj,
        exposed: bool,
    ) -> tuple[ColumnBase, ...]:
        """
        Extract the children columns from a pylibcudf.Column.

        ColumnBase currently assumes children are also ColumnBase objects.

        Parameters
        ----------
        plc_column : plc.Column
            The pylibcudf.Column to extract the children columns from.
        exposed : bool
            Whether the children columns are exposed.

        Returns
        -------
        tuple[ColumnBase, ...]
            The children columns.
        """
        return tuple(
            type(self).from_pylibcudf(child, data_ptr_exposed=exposed)
            for child in plc_column.children()
        )

    @property
    def _PANDAS_NA_VALUE(self):
        """Return appropriate NA value based on dtype."""
        if cudf.get_option("mode.pandas_compatible"):
            # In pandas compatibility mode, return pd.NA for all
            # nullable extension dtypes
            if is_pandas_nullable_extension_dtype(self.dtype):
                return self.dtype.na_value
            elif (
                self.dtype.kind == "f"
                and not is_pandas_nullable_extension_dtype(self.dtype)
            ):
                # For float dtypes, return np.nan
                return np.nan
            elif cudf.api.types.is_string_dtype(self.dtype):
                # numpy string dtype case, may be moved
                # to `StringColumn` later
                return None
        return pd.NA

    @property
    def base_size(self) -> int:
        return int(self.base_data.size / self.dtype.itemsize)  # type: ignore[union-attr]

    @property
    def dtype(self) -> DtypeObj:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def base_data(self) -> None | Buffer:
        return self._base_data

    @property
    def data(self) -> None | Buffer:
        if self.base_data is None:
            return None
        if self._data is None:
            start = self.offset * self.dtype.itemsize
            end = start + self.size * self.dtype.itemsize
            self._data = self.base_data[start:end]  # type: ignore[assignment]
        return self._data

    @property
    def data_ptr(self) -> int:
        if self.data is None:
            return 0
        else:
            # Save the original ptr
            original_ptr = self.data.get_ptr(mode="read")

            # Get the pointer which may trigger a copy due to copy-on-write
            ptr = self.data.get_ptr(mode="write")

            # Check if a new buffer was created or if the underlying data was modified
            # This happens both when the buffer object is replaced and when
            # ExposureTrackedBuffer.make_single_owner_inplace() is called
            if cudf.get_option("copy_on_write") and (ptr != original_ptr):
                # Update base_data to match the new data buffer
                self.set_base_data(self.data)

            return ptr

    def set_base_data(self, value: None | Buffer) -> None:
        if value is not None and not isinstance(value, Buffer):
            raise TypeError(
                "Expected a Buffer or None for data, "
                f"got {type(value).__name__}"
            )

        self._data = None
        self._base_data = value

    @property
    def nullable(self) -> bool:
        return self.base_mask is not None

    def has_nulls(self, include_nan: bool = False) -> bool:
        return int(self.null_count) != 0

    @property
    def base_mask(self) -> None | Buffer:
        return self._base_mask

    @property
    def mask(self) -> None | Buffer:
        if self._mask is None:
            if self.base_mask is None or self.offset == 0:
                self._mask = self.base_mask  # type: ignore[assignment]
            else:
                with acquire_spill_lock():
                    self._mask = as_buffer(  # type: ignore[assignment]
                        plc.null_mask.copy_bitmask(
                            self.to_pylibcudf(mode="read")
                        )
                    )
        return self._mask

    @property
    def mask_ptr(self) -> int:
        if self.mask is None:
            return 0
        else:
            # Save the original ptr
            original_ptr = self.mask.get_ptr(mode="read")

            # Get the pointer which may trigger a copy due to copy-on-write
            ptr = self.mask.get_ptr(mode="write")

            # Check if a new buffer was created or if the underlying data was modified
            # This happens both when the buffer object is replaced and when
            # ExposureTrackedBuffer.make_single_owner_inplace() is called
            if cudf.get_option("copy_on_write") and (ptr != original_ptr):
                # Update base_data to match the new data buffer
                self.set_base_mask(self.mask)

            return ptr

    def set_base_mask(self, value: None | Buffer) -> None:
        """
        Replaces the base mask buffer of the column inplace. This does not
        modify size or offset in any way, so the passed mask is expected to be
        compatible with the current offset.
        """
        if value is not None and not isinstance(value, Buffer):
            raise TypeError(
                "Expected a Buffer or None for mask, "
                f"got {type(value).__name__}"
            )

        if value is not None:
            # bitmask size must be relative to offset = 0 data.
            required_size = plc.null_mask.bitmask_allocation_size_bytes(
                self.base_size
            )
            if value.size < required_size:
                error_msg = (
                    "The Buffer for mask is smaller than expected, "
                    f"got {value.size} bytes, expected {required_size} bytes."
                )
                if self.offset > 0 or self.size < self.base_size:
                    error_msg += (
                        "\n\nNote: The mask is expected to be sized according "
                        "to the base allocation as opposed to the offsetted or"
                        " sized allocation."
                    )
                raise ValueError(error_msg)

        self._mask = None
        self._children = None
        self._base_mask = value  # type: ignore[assignment]
        self._clear_cache()

    def _clear_cache(self) -> None:
        self._distinct_count.clear()
        attrs = (
            "memory_usage",
            "is_monotonic_increasing",
            "is_monotonic_decreasing",
        )
        for attr in attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                # attr was not called yet, so ignore.
                pass
        self._null_count = None  # type: ignore[assignment]

    def set_mask(self, mask: Buffer | None) -> Self:
        """
        Replaces the mask buffer of the column and returns a new column.
        The input mask is assumed to be of appropriate size for self.
        """
        if isinstance(mask, Buffer):
            new_mask = plc.gpumemoryview(mask)
            new_null_count = plc.null_mask.null_count(
                new_mask,
                0,
                self.size,
            )
        elif mask is None:
            new_mask = None
            new_null_count = 0
        else:
            raise ValueError(
                f"Expected a Buffer object or None for mask, got {type(mask).__name__}"
            )
        new_plc_column = self.to_pylibcudf(
            mode="read", use_base=False
        ).with_mask(new_mask, new_null_count)
        return (
            type(self)
            .from_pylibcudf(  # type: ignore[return-value]
                new_plc_column,
            )
            ._with_type_metadata(self.dtype)
        )

    @property
    def null_count(self) -> int:
        if self._null_count is None:
            if not self.nullable or self.size == 0:
                self._null_count = 0
            else:
                with acquire_spill_lock():
                    self._null_count = plc.null_mask.null_count(
                        plc.gpumemoryview(self.base_mask),
                        self.offset,
                        self.offset + self.size,
                    )
        return self._null_count

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def base_children(self) -> tuple[ColumnBase, ...]:
        return self._base_children

    @property
    def children(self) -> tuple[ColumnBase, ...]:
        if self.offset == 0 and self.size == self.base_size:
            self._children = self.base_children  # type: ignore[assignment]
        if self._children is None:
            if not self.base_children:
                self._children = ()  # type: ignore[assignment]
            else:
                # Compute children from the column view (children factoring self.size)
                children = ColumnBase.from_pylibcudf(
                    self.to_pylibcudf(mode="read").copy()
                ).base_children
                dtypes = (
                    base_child.dtype for base_child in self.base_children
                )
                self._children = tuple(  # type: ignore[assignment]
                    child._with_type_metadata(dtype)
                    for child, dtype in zip(children, dtypes, strict=True)
                )
        return self._children  # type: ignore[return-value]

    def set_base_children(self, value: tuple[ColumnBase, ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError(
                f"Expected a tuple of Columns for children, got {type(value).__name__}"
            )
        if any(not isinstance(child, ColumnBase) for child in value):
            raise TypeError("All children must be Columns.")

        self._children = None
        self._base_children = value

    def _mimic_inplace(
        self, other_col: Self, inplace: bool = False
    ) -> None | Self:
        """
        Given another column, update the attributes of this column to mimic an
        inplace operation. This does not modify the memory of Buffers, but
        instead replaces the Buffers and other attributes underneath the column
        object with the Buffers and attributes from the other column.
        """
        if inplace:
            self._offset = other_col.offset
            self._size = other_col.size
            self._dtype = other_col._dtype
            self.set_base_data(other_col.base_data)
            self.set_base_children(other_col.base_children)
            self.set_base_mask(other_col.base_mask)
            # TODO: self._clear_cache here?
            return None
        else:
            return other_col

    # TODO: Consider whether this function should support some sort of `copy`
    # parameter. Not urgent until this functionality is moved up to the Frame
    # layer and made public. This function will also need to mark the
    # underlying buffers as exposed before this function can itself be exposed
    # publicly.  User requests to convert to pylibcudf must assume that the
    # data may be modified afterwards.
    def to_pylibcudf(
        self, mode: Literal["read", "write"], *, use_base: bool = True
    ) -> plc.Column:
        """Convert this Column to a pylibcudf.Column.

        This function will generate a pylibcudf Column pointing to the same
        data, mask, and children as this one.

        Parameters
        ----------
        mode : str
            Supported values are {"read", "write"} If "write", the data pointed
            to may be modified by the caller. If "read", the data pointed to
            must not be modified by the caller.  Failure to fulfill this
            contract will cause incorrect behavior.
        use_base : bool, default True
            Whether to use the column's base data, mask, and children,
            or data, mask, and children relative to a 0 offset.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column referencing the same data.
        """
        # TODO: Categoricals will need to be treated differently eventually.
        # There is no 1-1 correspondence between cudf and libcudf for
        # categoricals because cudf supports ordered and unordered categoricals
        # while libcudf supports only unordered categoricals (see
        # https://github.com/rapidsai/cudf/pull/8567).
        if isinstance(self.dtype, cudf.CategoricalDtype):
            col = self.base_children[0]
        else:
            col = self

        dtype = dtype_to_pylibcudf_type(col.dtype)

        data = None
        if col.base_data is not None:
            if use_base:
                data_buff = col.base_data
            else:
                data_buff = col.data  # type: ignore[assignment]
            cai = cuda_array_interface_wrapper(
                ptr=data_buff.get_ptr(mode=mode),
                size=data_buff.size,
                owner=data_buff,
            )
            data = plc.gpumemoryview(cai)

        mask = None
        if self.nullable:
            # TODO: Are we intentionally use self's mask instead of col's?
            # Where is the mask stored for categoricals?
            if use_base:
                mask_buff = self.base_mask
            else:
                mask_buff = self.mask
            cai = cuda_array_interface_wrapper(
                ptr=mask_buff.get_ptr(mode=mode),  # type: ignore[union-attr]
                size=mask_buff.size,  # type: ignore[union-attr]
                owner=mask_buff,
            )
            mask = plc.gpumemoryview(cai)

        children = []
        if col.base_children:
            children = [
                child_column.to_pylibcudf(mode=mode, use_base=use_base)
                for child_column in (
                    col.base_children if use_base else col.children
                )
            ]

        return plc.Column(
            dtype,
            self.size,
            data,
            mask,
            self.null_count,
            self.offset if use_base else 0,
            children,
        )

    @classmethod
    def from_pylibcudf(
        cls, col: plc.Column, data_ptr_exposed: bool = False
    ) -> Self:
        """Create a Column from a pylibcudf.Column.

        This function will generate a Column pointing to the provided pylibcudf
        Column.  It will directly access the data and mask buffers of the
        pylibcudf Column, so the newly created object is not tied to the
        lifetime of the original pylibcudf.Column.

        Parameters
        ----------
        col : pylibcudf.Column
            The object to copy.
        data_ptr_exposed : bool
            Whether the data buffer is exposed.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column referencing the same data.
        """
        if col.type().id() == plc.TypeId.TIMESTAMP_DAYS:
            col = plc.unary.cast(
                col, plc.DataType(plc.TypeId.TIMESTAMP_SECONDS)
            )
        elif col.type().id() == plc.TypeId.EMPTY:
            new_dtype = plc.DataType(plc.TypeId.INT8)

            col = plc.column_factories.make_numeric_column(
                new_dtype, col.size(), plc.types.MaskState.ALL_NULL
            )

        dtype = dtype_from_pylibcudf_column(col)

        return build_column(  # type: ignore[return-value]
            plc_column=col,
            size=col.size(),
            dtype=dtype,
            offset=col.offset(),
            null_count=col.null_count(),
            exposed=data_ptr_exposed,
        )

    @classmethod
    def from_cuda_array_interface(cls, arbitrary: Any) -> ColumnBase:
        """
        Create a Column from an object implementing the CUDA array interface.

        Parameters
        ----------
        arbitrary : Any
            The object to convert.

        Returns
        -------
        Column
        """
        if (
            cai := getattr(arbitrary, "__cuda_array_interface__", None)
        ) is None:
            raise ValueError(
                "Object does not implement __cuda_array_interface__"
            )

        cai_dtype = np.dtype(cai["typestr"])
        check_invalid_array(cai["shape"], cai_dtype)
        arbitrary = maybe_reshape(
            arbitrary, cai["shape"], cai["strides"], cai_dtype
        )

        # TODO: Can remove once from_cuda_array_interface can handle masks
        # https://github.com/rapidsai/cudf/issues/19122
        if (mask := cai.get("mask", None)) is not None:
            cai_copy = cai.copy()
            cai_copy.pop("mask")
            arbitrary = SimpleNamespace(__cuda_array_interface__=cai_copy)
        else:
            mask = None

        column = ColumnBase.from_pylibcudf(
            plc.Column.from_cuda_array_interface(arbitrary),
            data_ptr_exposed=cudf.get_option("copy_on_write"),
        )
        if mask is not None:
            cai_mask = mask.__cuda_array_interface__
            if cai_mask["typestr"][1] == "t":
                mask_size = plc.null_mask.bitmask_allocation_size_bytes(
                    cai_mask["shape"][0]
                )
                mask_buff = as_buffer(
                    data=cai_mask["data"][0], size=mask_size, owner=mask
                )
            elif cai_mask["typestr"][1] == "b":
                mask_buff = ColumnBase.from_cuda_array_interface(
                    mask,
                ).as_mask()
            else:
                mask_buff = as_buffer(mask)
            required_num_bytes = -(-column.size // 8)  # ceiling divide
            if mask_buff.size < required_num_bytes:
                raise ValueError(
                    f"The value for mask is smaller than expected, got {mask.size} bytes, "
                    f"expected {required_num_bytes} bytes."
                )
            column = column.set_mask(mask_buff)
        return column

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"{object.__repr__(self)}\n"
            f"{self.to_arrow().to_string()}\n"
            f"dtype: {self.dtype}"
        )

    def _prep_pandas_compat_repr(self) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        if self.has_nulls():
            return self.astype(np.dtype("str")).fillna(
                str(self._PANDAS_NA_VALUE)
            )
        return self

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        """Convert object to pandas type.

        The default implementation falls back to PyArrow for the conversion.
        """
        # This default implementation does not handle nulls in any meaningful
        # way
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        pa_array = self.to_arrow()

        # Check if dtype is an ArrowDtype or pd.ExtensionDtype subclass
        if arrow_type or (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(self.dtype, pd.ArrowDtype)
        ):
            return pd.Index(pd.arrays.ArrowExtensionArray(pa_array))
        elif (
            nullable
            or (
                cudf.get_option("mode.pandas_compatible")
                and is_pandas_nullable_extension_dtype(self.dtype)
            )
        ) and is_pandas_nullable_extension_dtype(
            pandas_nullable_dtype := np_dtypes_to_pandas_dtypes.get(
                self.dtype, self.dtype
            )
        ):
            pandas_array = pandas_nullable_dtype.__from_arrow__(pa_array)
            return pd.Index(pandas_array, copy=False)
        else:
            return pd.Index(pa_array.to_pandas())

    @property
    def values_host(self) -> np.ndarray:
        """
        Return a numpy representation of the Column.
        """
        return self.to_pandas().to_numpy()

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the Column.
        """
        raise NotImplementedError(f"cupy does not support {self.dtype}")

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        raise NotImplementedError

    @acquire_spill_lock()
    def clip(self, lo: ScalarLike, hi: ScalarLike) -> Self:
        plc_column = plc.replace.clamp(
            self.to_pylibcudf(mode="read"),
            pa_scalar_to_plc_scalar(
                pa.scalar(lo, type=cudf_dtype_to_pa_type(self.dtype))
            ),
            pa_scalar_to_plc_scalar(
                pa.scalar(hi, type=cudf_dtype_to_pa_type(self.dtype))
            ),
        )
        return type(self).from_pylibcudf(plc_column)

    def equals(self, other: ColumnBase, check_dtypes: bool = False) -> bool:
        if not isinstance(other, ColumnBase) or len(self) != len(other):
            return False
        elif self is other:
            return True
        elif check_dtypes and self.dtype != other.dtype:
            return False
        elif not check_dtypes and not _can_values_be_equal(
            self.dtype, other.dtype
        ):
            return False
        elif self.null_count != other.null_count:
            return False
        ret = self._binaryop(other, "NULL_EQUALS")
        if ret is NotImplemented:
            return False
        return ret.all()

    def all(self, skipna: bool = True) -> bool:
        # The skipna argument is only used for numerical columns.
        # If all entries are null the result is True, including when the column
        # is empty.
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        if self.null_count == self.size:
            return True
        return bool(self.reduce("all"))

    def any(self, skipna: bool = True) -> bool:
        # Early exit for fast cases.
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        if not skipna and self.has_nulls():
            return True
        elif skipna and self.null_count == self.size:
            return False
        return self.reduce("any")

    def dropna(self) -> Self:
        if self.has_nulls():
            return ColumnBase.from_pylibcudf(
                stream_compaction.drop_nulls([self])[0]
            )._with_type_metadata(self.dtype)  # type: ignore[return-value]
        else:
            return self.copy()

    @acquire_spill_lock()
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
        return self.to_pylibcudf(mode="read").to_arrow()

    @classmethod
    def from_arrow(cls, array: pa.Array | pa.ChunkedArray) -> ColumnBase:
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
        [
          1,
          2,
          3,
          4
        ]
        dtype: int8
        """
        if not isinstance(array, (pa.Array, pa.ChunkedArray)):
            raise TypeError("array should be PyArrow array or chunked array")
        elif pa.types.is_float16(array.type):
            raise NotImplementedError(
                "Type casting from `float16` to `float32` is not "
                "yet supported in pyarrow, see: "
                "https://github.com/apache/arrow/issues/20213"
            )
        elif isinstance(array.type, ArrowIntervalType):
            return cudf.core.column.IntervalColumn.from_arrow(array)
        elif pa.types.is_null(array.type):
            # default "empty" type
            array = array.cast(pa.string())

        if pa.types.is_dictionary(array.type):
            if isinstance(array, pa.Array):
                dict_array = cast(pa.DictionaryArray, array)
                codes: pa.Array | pa.ChunkedArray = dict_array.indices
                dictionary: pa.Array | pa.ChunkedArray = dict_array.dictionary
            else:
                codes = pa.chunked_array(
                    [
                        cast(pa.DictionaryArray, chunk).indices
                        for chunk in array.chunks
                    ],
                    type=array.type.index_type,
                )
                dictionary = pc.unique(
                    pa.chunked_array(
                        [
                            cast(pa.DictionaryArray, chunk).dictionary
                            for chunk in array.chunks
                        ],
                        type=array.type.value_type,
                    )
                )
            with acquire_spill_lock():
                if version.parse(pa.__version__) < version.parse(
                    "16"
                ) and isinstance(codes, pa.ChunkedArray):
                    result = cls.from_pylibcudf(
                        plc.Table.from_arrow(
                            pa.table({None: codes})
                        ).columns()[0]
                    )
                    categories = cls.from_pylibcudf(
                        plc.Table.from_arrow(
                            pa.table({None: dictionary})
                        ).columns()[0]
                    )
                else:
                    result = cls.from_pylibcudf(plc.Column.from_arrow(codes))
                    categories = cls.from_pylibcudf(
                        plc.Column.from_arrow(dictionary)
                    )
            return result._with_type_metadata(
                CategoricalDtype(
                    categories=categories, ordered=array.type.ordered
                )
            )
        else:
            if version.parse(pa.__version__) < version.parse(
                "16"
            ) and isinstance(array, pa.ChunkedArray):
                result = cls.from_pylibcudf(
                    plc.Table.from_arrow(pa.table({None: array})).columns()[0]
                )
            else:
                result = cls.from_pylibcudf(plc.Column.from_arrow(array))
            return result._with_type_metadata(
                cudf_dtype_from_pa_type(array.type)
            )

    @acquire_spill_lock()
    def _get_mask_as_column(self) -> ColumnBase:
        plc_column = plc.transform.mask_to_bools(
            self.base_mask.get_ptr(mode="read"),  # type: ignore[union-attr]
            self.offset,
            self.offset + len(self),
        )
        return type(self).from_pylibcudf(plc_column)

    @cached_property
    def memory_usage(self) -> int:
        n = 0
        if self.data is not None:
            n += self.data.size
        if self.nullable:
            n += plc.null_mask.bitmask_allocation_size_bytes(self.size)
        return n

    def _fill(
        self,
        fill_value: plc.Scalar,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Self | None:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        if not inplace or self.dtype == CUDF_STRING_DTYPE:
            with acquire_spill_lock():
                result = type(self).from_pylibcudf(
                    plc.filling.fill(
                        self.to_pylibcudf(mode="read"),
                        begin,
                        end,
                        fill_value,
                    )
                )
            if self.dtype == CUDF_STRING_DTYPE:
                return self._mimic_inplace(result, inplace=True)
            return result

        if not fill_value.is_valid() and not self.nullable:
            mask = as_buffer(
                plc.null_mask.create_null_mask(
                    self.size, plc.types.MaskState.ALL_VALID
                )
            )
            self.set_base_mask(mask)

        with acquire_spill_lock():
            plc.filling.fill_in_place(
                self.to_pylibcudf(mode="write"),
                begin,
                end,
                fill_value,
            )
        return self

    @acquire_spill_lock()
    def shift(self, offset: int, fill_value: ScalarLike) -> Self:
        plc_fill_value = self._scalar_to_plc_scalar(fill_value)
        plc_col = plc.copying.shift(
            self.to_pylibcudf(mode="read"),
            offset,
            plc_fill_value,
        )
        return type(self).from_pylibcudf(plc_col)

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
            with acquire_spill_lock():
                result = type(self).from_pylibcudf(
                    self.to_pylibcudf(mode="read").copy()
                )
            return result._with_type_metadata(self.dtype)  # type: ignore[return-value]
        else:
            col = type(self)(
                plc_column=self.plc_column,
                size=self.size,
                dtype=self.dtype,
                offset=self.offset,
                null_count=self.null_count,
                exposed=False,
            )
            # copy-on-write and spilling logic tracked on the Buffers
            # so copy over the Buffers from self
            col.set_base_children(
                tuple(child.copy(deep=False) for child in self.base_children)
            )
            col.set_base_data(
                self.base_data.copy(deep=False)
                if self.base_data is not None
                else None
            )
            col.set_base_mask(
                self.base_mask.copy(deep=False)
                if self.base_mask is not None
                else None
            )
            return col

    def element_indexing(self, index: int):
        """Default implementation for indexing to an element

        Raises
        ------
        ``IndexError`` if out-of-bound

        Notes
        -----
        Subclass should override this method to not return a pyarrow.Scalar
        (May not be needed once pylibcudf.Scalar.as_py() exists.)
        """
        if index < 0:
            index = len(self) + index
        if index > len(self) - 1 or index < 0:
            raise IndexError("single positional indexer is out-of-bounds")
        with acquire_spill_lock():
            plc_scalar = plc.copying.get_element(
                self.to_pylibcudf(mode="read"),
                index,
            )
        py_element = plc_scalar.to_arrow()
        if not py_element.is_valid:
            return self._PANDAS_NA_VALUE
        # Calling .as_py() on a pyarrow.StructScalar with duplicate field names
        # would raise. So we need subclasses to convert handle pyarrow scalars
        # manually
        return py_element

    def slice(self, start: int, stop: int, stride: int | None = None) -> Self:
        stride = 1 if stride is None else stride
        if start < 0:
            start = start + len(self)
        if stop < 0 and not (stride < 0 and stop == -1):
            stop = stop + len(self)
        if (stride > 0 and start >= stop) or (stride < 0 and start <= stop):
            return cast(Self, column_empty(0, self.dtype))
        # compute mask slice
        if stride == 1:
            with acquire_spill_lock():
                result = [
                    type(self).from_pylibcudf(col)
                    for col in plc.copying.slice(
                        self.to_pylibcudf(mode="read"),
                        [start, stop],
                    )
                ]
            return result[0]._with_type_metadata(self.dtype)  # type: ignore[return-value]
        else:
            # Need to create a gather map for given slice with stride
            gather_map = as_column(
                range(start, stop, stride),
                dtype=np.dtype(np.int32),
            )
            return self.take(gather_map)

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if is_scalar(value):
            if value is cudf.NA:
                value = None
            try:
                pa_scalar = pa.scalar(
                    value, type=cudf_dtype_to_pa_type(self.dtype)
                )
            except ValueError as err:
                raise TypeError(
                    f"Cannot set value of type {type(value)} to column of type {self.dtype}"
                ) from err
            return pa_scalar_to_plc_scalar(pa_scalar)
        else:
            return as_column(value, dtype=self.dtype)

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set the value of ``self[key]`` to ``value``.

        If ``value`` and ``self`` are of different types, ``value`` is coerced
        to ``self.dtype``. Assumes ``self`` and ``value`` are index-aligned.
        """
        value_normalized = self._cast_setitem_value(value)
        if isinstance(key, slice):
            out: ColumnBase | None = self._scatter_by_slice(
                key, value_normalized
            )
        else:
            key = as_column(key)
            if len(key) == 0:
                key = key.astype(SIZE_TYPE_DTYPE)
            if not is_dtype_obj_numeric(key.dtype):
                raise ValueError(f"Invalid scatter map type {key.dtype}.")
            out = self._scatter_by_column(key, value_normalized)

        if out:
            self._mimic_inplace(out, inplace=True)

    def _all_bools_with_nulls(
        self, other: ColumnBase, bool_fill_value: bool
    ) -> ColumnBase:
        # Might be able to remove if we share more of
        # DatetimeColumn._binaryop & TimedeltaColumn._binaryop
        if self.has_nulls() and other.has_nulls():
            result_mask = (
                self._get_mask_as_column() & other._get_mask_as_column()
            )
        elif self.has_nulls():
            result_mask = self._get_mask_as_column()
        elif other.has_nulls():
            result_mask = other._get_mask_as_column()
        else:
            result_mask = None

        result_col = as_column(
            bool_fill_value, dtype=np.dtype(np.bool_), length=len(self)
        )
        if result_mask is not None:
            result_col = result_col.set_mask(result_mask.as_mask())
        return result_col

    def _scatter_by_slice(
        self,
        key: builtins.slice,
        value: plc.Scalar | ColumnBase,
    ) -> Self | None:
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
            self.dtype, (cudf.StructDtype, cudf.ListDtype)
        ):
            # NOTE: List & Struct dtypes aren't supported by both
            # inplace & out-of-place fill. Hence we need to use scatter for
            # these two types.
            if isinstance(value, plc.Scalar):
                return self._fill(value, start, stop, inplace=True)
            else:
                with acquire_spill_lock():
                    return type(self).from_pylibcudf(
                        plc.copying.copy_range(
                            value.to_pylibcudf(mode="read"),
                            self.to_pylibcudf(mode="read"),
                            0,
                            num_keys,
                            start,
                        )
                    )

        # step != 1, create a scatter map with arange
        scatter_map = cast(
            cudf.core.column.NumericalColumn,
            as_column(
                rng,
                dtype=np.dtype(np.int32),
            ),
        )

        return self._scatter_by_column(scatter_map, value)

    def _scatter_by_column(
        self,
        key: NumericalColumn,
        value: plc.Scalar | ColumnBase,
        bounds_check: bool = True,
    ) -> Self:
        if key.dtype.kind == "b":
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

        if key.dtype.kind == "b":
            with acquire_spill_lock():
                plc_table = plc.copying.boolean_mask_scatter(
                    plc.Table([value.to_pylibcudf(mode="read")])
                    if isinstance(value, ColumnBase)
                    else [value],
                    plc.Table([self.to_pylibcudf(mode="read")]),
                    key.to_pylibcudf(mode="read"),
                )
                return (
                    type(self)  # type: ignore[return-value]
                    .from_pylibcudf(plc_table.columns()[0])
                    ._with_type_metadata(self.dtype)
                )
        else:
            return ColumnBase.from_pylibcudf(  # type: ignore[return-value]
                copying.scatter(
                    cast(list[plc.Scalar], [value])
                    if isinstance(value, plc.Scalar)
                    else cast(list[ColumnBase], [value]),
                    key,
                    [self],
                    bounds_check=bounds_check,
                )[0]
            )._with_type_metadata(self.dtype)

    def _check_scatter_key_length(
        self, num_keys: int, value: plc.Scalar | ColumnBase
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

    def _scalar_to_plc_scalar(self, scalar: ScalarLike) -> plc.Scalar:
        """Return a pylibcudf.Scalar that matches the type of self.dtype"""
        if not isinstance(scalar, pa.Scalar):
            scalar = pa.scalar(scalar)
        return pa_scalar_to_plc_scalar(
            scalar.cast(cudf_dtype_to_pa_type(self.dtype))
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            return self._scalar_to_plc_scalar(fill_value)
        return as_column(fill_value).astype(self.dtype)

    @acquire_spill_lock()
    def replace(
        self, values_to_replace: Self, replacement_values: Self
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.replace.find_and_replace_all(
                self.to_pylibcudf(mode="read"),
                values_to_replace.to_pylibcudf(mode="read"),
                replacement_values.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def repeat(self, repeats: int) -> Self:
        return type(self).from_pylibcudf(
            plc.filling.repeat(
                plc.Table([self.to_pylibcudf(mode="read")]), repeats
            ).columns()[0]
        )

    def fillna(
        self,
        fill_value: ScalarLike | ColumnLike,
        method: Literal["ffill", "bfill", None] = None,
    ) -> Self:
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if not self.has_nulls(include_nan=True):
            return self.copy()
        elif method is None:
            if is_scalar(fill_value) and _is_null_host_scalar(fill_value):
                return self.copy()
            else:
                fill_value = self._validate_fillna_value(fill_value)

        if fill_value is None and method is None:
            raise ValueError("Must specify a fill 'value' or 'method'.")

        if fill_value and method:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        input_col = self.nans_to_nulls()

        with acquire_spill_lock():
            plc_replace: plc.replace.ReplacePolicy | plc.Scalar
            if method:
                plc_replace = (
                    plc.replace.ReplacePolicy.PRECEDING
                    if method == "ffill"
                    else plc.replace.ReplacePolicy.FOLLOWING
                )
            elif isinstance(fill_value, plc.Scalar):
                plc_replace = fill_value
            else:
                plc_replace = fill_value.to_pylibcudf(mode="read")
            plc_column = plc.replace.replace_nulls(
                input_col.to_pylibcudf(mode="read"),
                plc_replace,
            )
            result = type(self).from_pylibcudf(plc_column)
        return result._with_type_metadata(self.dtype)  # type: ignore[return-value]

    @acquire_spill_lock()
    def is_valid(self) -> ColumnBase:
        """Identify non-null values"""
        return type(self).from_pylibcudf(
            plc.unary.is_valid(self.to_pylibcudf(mode="read"))
        )

    def isnan(self) -> ColumnBase:
        """Identify NaN values in a Column."""
        if self.dtype.kind != "f":
            return as_column(False, length=len(self))
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.unary.is_nan(self.to_pylibcudf(mode="read"))
            )

    def notnan(self) -> ColumnBase:
        """Identify non-NaN values in a Column."""
        if self.dtype.kind != "f":
            return as_column(True, length=len(self))
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.unary.is_not_nan(self.to_pylibcudf(mode="read"))
            )

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column."""
        if not self.has_nulls(include_nan=self.dtype.kind == "f"):
            return as_column(False, length=len(self))

        with acquire_spill_lock():
            result = type(self).from_pylibcudf(
                plc.unary.is_null(self.to_pylibcudf(mode="read"))
            )

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of a float column
            result = result | self.isnan()

        return result

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column."""
        if not self.has_nulls(include_nan=self.dtype.kind == "f"):
            return as_column(True, length=len(self))

        with acquire_spill_lock():
            result = type(self).from_pylibcudf(
                plc.unary.is_valid(self.to_pylibcudf(mode="read"))
            )

        if self.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of a float column
            result = result & self.notnan()

        return result

    @cached_property
    def nan_count(self) -> int:
        return 0

    def interpolate(self, index: Index) -> ColumnBase:
        # figure out where the nans are
        mask = self.isnull()

        # trivial cases, all nan or no nans
        if not mask.any() or mask.all():
            return self.copy()

        from cudf.core.index import RangeIndex

        valid_locs = ~mask
        if isinstance(index, RangeIndex):
            # Each point is evenly spaced, index values don't matter
            known_x = cp.flatnonzero(valid_locs.values)
        else:
            known_x = index._column.apply_boolean_mask(valid_locs).values
        known_y = self.apply_boolean_mask(valid_locs).values

        result = cp.interp(index.to_cupy(), known_x, known_y)

        first_nan_idx = valid_locs.values.argmax().item()
        result[:first_nan_idx] = np.nan
        return as_column(result)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
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
        if not is_scalar(value):
            raise ValueError("value must be a scalar")
        else:
            value = as_column(value, dtype=self.dtype, length=1)
        mask = value.contains(self)
        return as_column(
            range(len(self)), dtype=SIZE_TYPE_DTYPE
        ).apply_boolean_mask(mask)  # type: ignore[return-value]

    def _find_first_and_last(self, value: ScalarLike) -> tuple[int, int]:
        indices = self.indices_of(value)
        if n := len(indices):
            return (  # type: ignore[return-value]
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
        return concat_columns([self, other])

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase:
        raise TypeError(f"cannot perform quantile with type {self.dtype}")

    def take(
        self,
        indices: ColumnBase,
        *,
        nullify: bool = False,
        check_bounds: bool = True,
    ) -> Self:
        """Return Column by taking values from the corresponding *indices*.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        # Handle zero size
        if indices.size == 0:
            return cast(Self, column_empty(row_count=0, dtype=self.dtype))

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if indices.dtype.kind not in {"u", "i"}:
            indices = indices.astype(SIZE_TYPE_DTYPE)
        GatherMap(indices, len(self), nullify=not check_bounds or nullify)
        gathered = ColumnBase.from_pylibcudf(
            copying.gather([self], indices, nullify=nullify)[0]  # type: ignore[arg-type]
        )
        return gathered._with_type_metadata(self.dtype)  # type: ignore[return-value]

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
        lhs, rhs = self._process_values_for_isin(values)
        if lhs.dtype != rhs.dtype:
            if lhs.null_count and rhs.null_count:
                return lhs.isnull()
            else:
                return as_column(
                    False, length=len(self), dtype=np.dtype(np.bool_)
                )
        elif lhs.null_count == 0 and (rhs.null_count == len(rhs)):
            return as_column(False, length=len(self), dtype=np.dtype(np.bool_))

        result = rhs.contains(lhs)
        if lhs.null_count > 0:
            # If one of the needles is null, then the result contains
            # nulls, these nulls should be replaced by whether or not the
            # haystack contains a null.
            # TODO: this is unnecessary if we resolve
            # https://github.com/rapidsai/cudf/issues/14515 by
            # providing a mode in which cudf::contains does not mask
            # the result.
            result = result.fillna(rhs.null_count > 0)
        return result

    def _process_values_for_isin(
        self, values: Sequence
    ) -> tuple[ColumnBase, ColumnBase]:
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

    def as_mask(self) -> Buffer:
        """Convert booleans to bitmask

        Returns
        -------
        Buffer
        """
        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        with acquire_spill_lock():
            mask, _ = plc.transform.bools_to_mask(
                self.to_pylibcudf(mode="read")
            )
            return as_buffer(mask)

    @property
    def is_unique(self) -> bool:
        # distinct_count might already be cached
        return self.distinct_count(dropna=False) == len(self)

    @cached_property
    def is_monotonic_increasing(self) -> bool:
        return not self.has_nulls(include_nan=True) and sorting.is_sorted(
            [self], [True], ["first"]
        )

    @cached_property
    def is_monotonic_decreasing(self) -> bool:
        return not self.has_nulls(include_nan=True) and sorting.is_sorted(
            [self], [False], ["first"]
        )

    @acquire_spill_lock()
    def contains(self, other: ColumnBase) -> ColumnBase:
        """
        Check whether column contains multiple values.

        Parameters
        ----------
        other : Column
            A column of values to search for
        """
        return ColumnBase.from_pylibcudf(
            plc.search.contains(
                self.to_pylibcudf(mode="read"),
                other.to_pylibcudf(mode="read"),
            )
        )

    def sort_values(
        self: Self,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> Self:
        if (not ascending and self.is_monotonic_decreasing) or (
            ascending and self.is_monotonic_increasing
        ):
            return self.copy()
        order = sorting.ordering([ascending], [na_position])
        with acquire_spill_lock():
            plc_table = plc.sorting.sort(
                plc.Table([self.to_pylibcudf(mode="read")]),
                order[0],
                order[1],
            )
            return (
                type(self)  # type: ignore[return-value]
                .from_pylibcudf(plc_table.columns()[0])
                ._with_type_metadata(self.dtype)
            )

    def distinct_count(self, dropna: bool = True) -> int:
        try:
            return self._distinct_count[dropna]
        except KeyError:
            with acquire_spill_lock():
                result = plc.stream_compaction.distinct_count(
                    self.to_pylibcudf(mode="read"),
                    plc.types.NullPolicy.EXCLUDE
                    if dropna
                    else plc.types.NullPolicy.INCLUDE,
                    plc.types.NanPolicy.NAN_IS_NULL
                    if dropna
                    else plc.types.NanPolicy.NAN_IS_VALID,
                )
            self._distinct_count[dropna] = result
            return self._distinct_count[dropna]

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        raise NotImplementedError()

    @acquire_spill_lock()
    def cast(self, dtype: DtypeObj) -> ColumnBase:
        result = type(self).from_pylibcudf(
            plc.unary.cast(
                self.to_pylibcudf(mode="read"), dtype_to_pylibcudf_type(dtype)
            )
        )
        if isinstance(
            result.dtype,
            (cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype),
        ):
            if cudf.get_option("mode.pandas_compatible") and not isinstance(
                dtype, DecimalDtype
            ):
                result._dtype = dtype
            else:
                result.dtype.precision = dtype.precision  # type: ignore[union-attr]
        if cudf.get_option("mode.pandas_compatible") and result.dtype != dtype:
            result._dtype = dtype
        return result

    def astype(self, dtype: DtypeObj, copy: bool | None = False) -> ColumnBase:
        if self.dtype == dtype:
            result = self
        elif len(self) == 0:
            result = column_empty(0, dtype=dtype)
        else:
            if isinstance(dtype, CategoricalDtype):
                result = self.as_categorical_column(dtype)
            elif is_dtype_obj_interval(dtype):
                result = self.as_interval_column(dtype)  # type: ignore[arg-type]
            elif is_dtype_obj_list(dtype) or is_dtype_obj_struct(dtype):
                if self.dtype != dtype:
                    raise NotImplementedError(
                        f"Casting {self.dtype} columns not currently supported"
                    )
                result = self
            elif is_dtype_obj_decimal(dtype):
                result = self.as_decimal_column(dtype)  # type: ignore[arg-type]
            elif dtype.kind == "M":
                result = self.as_datetime_column(dtype)
            elif dtype.kind == "m":
                result = self.as_timedelta_column(dtype)
            elif dtype.kind in {"O", "U"}:
                if (
                    cudf.get_option("mode.pandas_compatible")
                    and isinstance(dtype, pd.ArrowDtype)
                    and not cudf.api.types.is_string_dtype(dtype)
                ):
                    raise TypeError(f"Unsupported dtype for astype: {dtype}")
                result = self.as_string_column(dtype)
            else:
                result = self.as_numerical_column(dtype)

        if copy and result is self:
            return result.copy(deep=copy)
        return result

    def as_categorical_column(
        self, dtype: CategoricalDtype
    ) -> CategoricalColumn:
        if dtype._categories is not None:
            # Re-label self w.r.t. the provided categories
            codes = self._label_encoding(cats=dtype._categories)
        else:
            # Compute categories from self
            cats = self.unique().sort_values()
            codes = self._label_encoding(cats=cats)
            if self.has_nulls():
                # TODO: Make dropna shallow copy if there are no nulls?
                cats = cats.dropna()
            dtype = CategoricalDtype(categories=cats, ordered=dtype.ordered)
        return codes.set_mask(self.mask)._with_type_metadata(dtype)  # type: ignore[return-value]

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        raise NotImplementedError

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        raise NotImplementedError

    def as_interval_column(self, dtype: IntervalDtype) -> IntervalColumn:
        raise NotImplementedError

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        raise NotImplementedError

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        raise NotImplementedError

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalBaseColumn:
        raise NotImplementedError

    def apply_boolean_mask(self, mask: ColumnBase) -> ColumnBase:
        if mask.dtype.kind != "b":
            raise ValueError("boolean_mask is not boolean type.")

        return ColumnBase.from_pylibcudf(
            stream_compaction.apply_boolean_mask([self], mask)[0]
        )._with_type_metadata(self.dtype)

    def argsort(
        self,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> NumericalColumn:
        if (ascending and self.is_monotonic_increasing) or (
            not ascending and self.is_monotonic_decreasing
        ):
            return cast(
                cudf.core.column.NumericalColumn, as_column(range(len(self)))
            )
        elif (ascending and self.is_monotonic_decreasing) or (
            not ascending and self.is_monotonic_increasing
        ):
            return cast(
                cudf.core.column.NumericalColumn,
                as_column(range(len(self) - 1, -1, -1)),
            )
        else:
            return ColumnBase.from_pylibcudf(  # type: ignore[return-value]
                sorting.order_by(
                    [self], [ascending], [na_position], stable=True
                )
            )

    def __arrow_array__(self, type: pa.DataType | None = None) -> None:
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
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

    def __invert__(self) -> ColumnBase:
        raise TypeError(
            f"Operation `~` not supported on {self.dtype.type.__name__}"
        )

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
        if is_pandas_nullable_extension_dtype(self.dtype) and self.has_nulls(
            include_nan=True
        ):
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        return ColumnBase.from_pylibcudf(
            sorting.search_sorted(  # type: ignore[return-value]
                [self],
                [value],
                side=side,
                ascending=[ascending],
                na_position=[na_position],
            )
        )

    def unique(self) -> Self:
        """
        Get unique values in the data
        """
        if self.is_unique:
            return self.copy()
        else:
            return ColumnBase.from_pylibcudf(
                stream_compaction.drop_duplicates([self], keep="first")[  # type: ignore[return-value]
                    0
                ]
            )._with_type_metadata(self.dtype)

    def serialize(self) -> tuple[dict, list]:
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

        header: dict[Any, Any] = {}
        frames = []
        try:
            dtype, dtype_frames = self.dtype.device_serialize()  # type: ignore[union-attr]
            header["dtype"] = dtype
            frames.extend(dtype_frames)
            header["dtype-is-cudf-serialized"] = True
        except AttributeError:
            if is_pandas_nullable_extension_dtype(self.dtype):
                header["dtype"] = pickle.dumps(self.dtype)
            else:
                header["dtype"] = self.dtype.str
            header["dtype-is-cudf-serialized"] = False

        if self.data is not None:
            data_header, data_frames = self.data.device_serialize()
            header["data"] = data_header
            frames.extend(data_frames)

        if self.mask is not None:
            mask_header, mask_frames = self.mask.device_serialize()
            header["mask"] = mask_header
            frames.extend(mask_frames)
        if self.children:
            child_headers, child_frames = zip(
                *(c.device_serialize() for c in self.children),
                strict=True,
            )
            header["subheaders"] = list(child_headers)
            frames.extend(chain(*child_frames))
        if isinstance(self.dtype, CategoricalDtype):
            header["codes_dtype"] = self.codes.dtype.str  # type: ignore[attr-defined]
        header["size"] = self.size
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> ColumnBase:
        def unpack(header, frames) -> tuple[Any, list]:
            count = header["frame_count"]
            obj = cls.device_deserialize(header, frames[:count])
            return obj, frames[count:]

        assert header["frame_count"] == len(frames), (
            f"Deserialization expected {header['frame_count']} frames, "
            f"but received {len(frames)}"
        )
        if header["dtype-is-cudf-serialized"]:
            dtype, frames = unpack(header["dtype"], frames)
        else:
            try:
                dtype = np.dtype(header["dtype"])
            except TypeError:
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
        if "codes_dtype" in header:
            codes_dtype = np.dtype(header["codes_dtype"])
        else:
            codes_dtype = None
        if mask is None:
            null_count = 0
        else:
            null_count = plc.null_mask.null_count(
                plc.gpumemoryview(mask), 0, header["size"]
            )
        if isinstance(dtype, IntervalDtype):
            # TODO: Handle in dtype_to_pylibcudf_type?
            plc_type = plc.DataType(plc.TypeId.STRUCT)
        else:
            plc_type = dtype_to_pylibcudf_type(
                codes_dtype if codes_dtype is not None else dtype
            )
        if isinstance(dtype, CategoricalDtype):
            data = children.pop(0)

        if isinstance(data, SpillableBuffer):
            data = spillable_gpumemoryview(data)
        elif data is not None:
            data = plc.gpumemoryview(data)
        if isinstance(mask, SpillableBuffer):
            mask = spillable_gpumemoryview(mask)
        elif mask is not None:
            mask = plc.gpumemoryview(mask)

        plc_column = plc.Column(
            plc_type,
            header["size"],
            data,
            mask,
            null_count,
            0,
            [child.to_pylibcudf(mode="read") for child in children],
        )
        return cls.from_pylibcudf(plc_column)._with_type_metadata(dtype)

    def unary_operator(self, unaryop: str) -> ColumnBase:
        raise TypeError(
            f"Operation {unaryop} not supported for dtype {self.dtype}."
        )

    def nans_to_nulls(self: Self) -> Self:
        """Convert NaN to NA."""
        return self

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
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
            return preprocessed.reduce(op, **kwargs)
        return preprocessed

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls(include_nan=False)

    def _process_for_reduction(
        self, skipna: bool = True, min_count: int = 0
    ) -> ColumnBase | ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        col = self.nans_to_nulls() if skipna else self
        if col.has_nulls():
            if skipna:
                col = col.dropna()
            else:
                return _get_nan_for_dtype(self.dtype)

        # TODO: If and when pandas decides to validate that `min_count` >= 0 we
        # should insert comparable behavior.
        # https://github.com/pandas-dev/pandas/issues/50022
        if min_count > 0:
            valid_count = len(col) - col.null_count
            if valid_count < min_count:
                return _get_nan_for_dtype(self.dtype)
        return col

    def _reduction_result_dtype(self, reduction_op: str) -> DtypeObj:
        """
        Determine the correct dtype to pass to libcudf based on
        the input dtype, data dtype, and specific reduction op
        """
        if reduction_op in {"any", "all"}:
            return np.dtype(np.bool_)
        return self.dtype

    def _with_type_metadata(self: ColumnBase, dtype: DtypeObj) -> ColumnBase:
        """
        Copies type metadata from self onto other, returning a new column.
        When ``self`` is a nested column, recursively apply this function on
        the children of ``self``.
        """
        # For Arrow dtypes, store them directly in the column's dtype property
        if isinstance(dtype, pd.ArrowDtype):
            self._dtype = cudf.dtype(dtype)
        return self

    def _label_encoding(
        self,
        cats: ColumnBase,
        dtype: DtypeObj | None = None,
    ) -> NumericalColumn:
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
        na_sentinel = pa.scalar(-1)

        def _return_sentinel_column():
            return as_column(na_sentinel, dtype=dtype, length=len(self))

        if dtype is None:
            dtype = min_signed_type(max(len(cats), na_sentinel.as_py()), 8)

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

        left_rows, right_rows = plc.join.left_join(
            plc.Table([self.to_pylibcudf(mode="read")]),
            plc.Table([cats.to_pylibcudf(mode="read")]),
            plc.types.NullEquality.EQUAL,
        )
        left_gather_map = type(self).from_pylibcudf(left_rows)
        right_gather_map = type(self).from_pylibcudf(right_rows)

        codes = as_column(range(len(cats)), dtype=dtype).take(
            right_gather_map, nullify=True
        )
        del right_gather_map
        del right_rows
        # reorder `codes` so that its values correspond to the
        # values of `self`:
        plc_codes = sorting.sort_by_key(
            [codes], [left_gather_map], [True], ["last"], stable=True
        )[0]
        return ColumnBase.from_pylibcudf(plc_codes).fillna(na_sentinel)  # type: ignore[return-value]

    @acquire_spill_lock()
    def copy_if_else(
        self, other: Self | plc.Scalar, boolean_mask: NumericalColumn
    ) -> Self:
        return (
            type(self)
            .from_pylibcudf(  # type: ignore[return-value]
                plc.copying.copy_if_else(
                    self.to_pylibcudf(mode="read"),
                    other
                    if isinstance(other, plc.Scalar)
                    else other.to_pylibcudf(mode="read"),
                    boolean_mask.to_pylibcudf(mode="read"),
                )
            )
            ._with_type_metadata(self.dtype)
        )

    def split_by_offsets(
        self, offsets: list[int]
    ) -> Generator[Self, None, None]:
        for cols in copying.columns_split([self], offsets):
            for col in cols:
                yield (  # type: ignore[misc]
                    type(self)
                    .from_pylibcudf(col)
                    ._with_type_metadata(self.dtype)
                )

    @acquire_spill_lock()
    def one_hot_encode(self, categories: ColumnBase) -> Generator[ColumnBase]:
        plc_table = plc.transform.one_hot_encode(
            self.to_pylibcudf(mode="read"),
            categories.to_pylibcudf(mode="read"),
        )
        return (
            type(self).from_pylibcudf(col, data_ptr_exposed=True)
            for col in plc_table.columns()
        )

    @acquire_spill_lock()
    def scan(self, scan_op: str, inclusive: bool, **kwargs) -> Self:
        return type(self).from_pylibcudf(
            plc.reduce.scan(
                self.to_pylibcudf(mode="read"),
                aggregation.make_aggregation(scan_op, kwargs).plc_obj,
                plc.reduce.ScanType.INCLUSIVE
                if inclusive
                else plc.reduce.ScanType.EXCLUSIVE,
            )
        )

    def reduce(self, reduction_op: str, **kwargs) -> ScalarLike:
        col_dtype = self._reduction_result_dtype(reduction_op)

        # check empty case
        if len(self) <= self.null_count:
            if reduction_op == "sum" or reduction_op == "sum_of_squares":
                return col_dtype.type(0)
            if reduction_op == "product":
                return col_dtype.type(1)
            if reduction_op == "any":
                return False

            return _get_nan_for_dtype(col_dtype)

        with acquire_spill_lock():
            plc_scalar = plc.reduce.reduce(
                self.to_pylibcudf(mode="read"),
                aggregation.make_aggregation(reduction_op, kwargs).plc_obj,
                dtype_to_pylibcudf_type(col_dtype),
            )
            result_col = type(self).from_pylibcudf(
                plc.Column.from_scalar(plc_scalar, 1)
            )
            if plc_scalar.type().id() in {
                plc.TypeId.DECIMAL128,
                plc.TypeId.DECIMAL64,
                plc.TypeId.DECIMAL32,
            }:
                scale = -plc_scalar.type().scale()
                # Narrow type for mypy - we know col_dtype is a decimal type from the check above
                assert isinstance(col_dtype, DecimalDtype)
                p = col_dtype.precision
                # https://docs.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql
                nrows = len(self)
                if reduction_op in {"min", "max"}:
                    new_p = p
                elif reduction_op == "sum":
                    new_p = p + nrows - 1
                elif reduction_op == "product":
                    new_p = p * nrows + nrows - 1
                elif reduction_op == "sum_of_squares":
                    new_p = 2 * p + nrows
                else:
                    raise NotImplementedError(
                        f"{reduction_op} not implemented for decimal types."
                    )
                precision = max(min(new_p, col_dtype.MAX_PRECISION), 0)
                new_dtype = type(col_dtype)(precision, scale)
                result_col = result_col.astype(new_dtype)
            elif isinstance(col_dtype, IntervalDtype):
                result_col = result_col._with_type_metadata(col_dtype)
        return result_col.element_indexing(0)

    @acquire_spill_lock()
    def minmax(self) -> tuple[ScalarLike, ScalarLike]:
        min_val, max_val = plc.reduce.minmax(self.to_pylibcudf(mode="read"))
        return (
            type(self)
            .from_pylibcudf(plc.Column.from_scalar(min_val, 1))
            .element_indexing(0),
            type(self)
            .from_pylibcudf(plc.Column.from_scalar(max_val, 1))
            .element_indexing(0),
        )

    @acquire_spill_lock()
    def rank(
        self,
        *,
        method: plc.aggregation.RankMethod,
        column_order: plc.types.Order,
        null_handling: plc.types.NullPolicy,
        null_precedence: plc.types.NullOrder,
        pct: bool,
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.sorting.rank(
                self.to_pylibcudf(mode="read"),
                method,
                column_order,
                null_handling,
                null_precedence,
                pct,
            )
        )

    @acquire_spill_lock()
    def label_bins(
        self,
        *,
        left_edge: Self,
        left_inclusive: bool,
        right_edge: Self,
        right_inclusive: bool,
    ) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.labeling.label_bins(
                self.to_pylibcudf(mode="read"),
                left_edge.to_pylibcudf(mode="read"),
                plc.labeling.Inclusive.YES
                if left_inclusive
                else plc.labeling.Inclusive.NO,
                right_edge.to_pylibcudf(mode="read"),
                plc.labeling.Inclusive.YES
                if right_inclusive
                else plc.labeling.Inclusive.NO,
            )
        )

    def _cast_self_and_other_for_where(
        self, other: ScalarLike | ColumnBase, inplace: bool
    ) -> tuple[ColumnBase, plc.Scalar | ColumnBase]:
        other_is_scalar = is_scalar(other)

        if other_is_scalar:
            if isinstance(other, (float, np.floating)) and not np.isnan(other):
                try:
                    is_safe = self.dtype.type(other) == other
                except OverflowError:
                    is_safe = False

                if not is_safe:
                    raise TypeError(
                        f"Cannot safely cast non-equivalent "
                        f"{type(other).__name__} to {self.dtype}"
                    )

            if is_na_like(other):
                if (
                    cudf.get_option("mode.pandas_compatible")
                    and not is_pandas_nullable_extension_dtype(self.dtype)
                    and self.dtype.kind not in {"i", "f", "u"}
                ):
                    raise MixedTypeError(
                        "Cannot use None or np.nan with non-Pandas nullable dtypes."
                    )
                return self, pa_scalar_to_plc_scalar(
                    pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
                )

        mixed_err = (
            "cudf does not support mixed types, please type-cast the column of "
            "dataframe/series and other to same dtypes."
        )

        if inplace:
            other_col = as_column(other)
            if is_mixed_with_object_dtype(other_col, self):
                raise TypeError(mixed_err)

            if other_col.dtype != self.dtype:
                try:
                    warn = (
                        find_common_type((other_col.dtype, self.dtype))
                        == CUDF_STRING_DTYPE
                    )
                except NotImplementedError:
                    warn = True
                if warn:
                    warnings.warn(
                        f"Type-casting from {other_col.dtype} "
                        f"to {self.dtype}, there could be potential data loss"
                    )
            other_out: plc.Scalar | ColumnBase
            if other_is_scalar:
                other_out = pa_scalar_to_plc_scalar(
                    pa.scalar(other, type=cudf_dtype_to_pa_type(self.dtype))
                )
            else:
                other_out = other_col.astype(self.dtype)
            return self, other_out

        if is_dtype_obj_numeric(
            self.dtype, include_decimal=False
        ) and as_column(other).can_cast_safely(self.dtype):
            common_dtype = self.dtype
        else:
            common_dtype = find_common_type(
                [
                    self.dtype,
                    np.min_scalar_type(other)
                    if other_is_scalar
                    else other.dtype,
                ]
            )
        other_col = as_column(other)
        if (
            is_mixed_with_object_dtype(other_col, self)
            or (self.dtype.kind == "b" and common_dtype.kind != "b")
            or (other_col.dtype.kind == "b" and self.dtype.kind != "b")
        ):
            raise TypeError(mixed_err)

        if other_is_scalar:
            other_out = pa_scalar_to_plc_scalar(
                pa.scalar(other, type=cudf_dtype_to_pa_type(common_dtype))
            )
        else:
            other_out = other.astype(common_dtype)

        return self.astype(common_dtype), other_out

    def where(
        self, cond: ColumnBase, other: ScalarLike | ColumnBase, inplace: bool
    ) -> ColumnBase:
        casted_col, casted_other = self._cast_self_and_other_for_where(
            other, inplace
        )
        return casted_col.copy_if_else(casted_other, cond)._with_type_metadata(  # type: ignore[arg-type]
            self.dtype
        )


def column_empty(
    row_count: int,
    dtype: DtypeObj = CUDF_STRING_DTYPE,
) -> ColumnBase:
    """
    Allocate a new column with the given row_count and dtype.

    * Passing row_count == 0 creates a size 0 column without a mask buffer.
    * Passing row_count > 0 creates an all null column with a mask buffer.

    Parameters
    ----------
    row_count : int
        Number of elements in the column.

    dtype : Dtype
        Type of the column.
    """
    if (is_struct := isinstance(dtype, StructDtype)) or isinstance(
        dtype, ListDtype
    ):
        if is_struct:
            children = tuple(
                column_empty(row_count, field_dtype)
                for field_dtype in dtype.fields.values()
            )
        else:
            children = (
                as_column(0, length=row_count + 1, dtype=SIZE_TYPE_DTYPE),
                column_empty(row_count, dtype=dtype.element_type),
            )
        mask = (
            None
            if row_count == 0
            else plc.gpumemoryview(
                plc.null_mask.create_null_mask(
                    row_count, plc.types.MaskState.ALL_NULL
                )
            )
        )
        return ColumnBase.from_pylibcudf(
            plc.Column(
                dtype_to_pylibcudf_type(dtype),
                row_count,
                None,
                mask,
                row_count,
                0,
                [child.to_pylibcudf(mode="read") for child in children],
            )
        )._with_type_metadata(dtype)
    else:
        if isinstance(dtype, CategoricalDtype):
            # May get downcast in _with_type_metadata
            plc_dtype = plc.DataType(plc.TypeId.INT64)
        else:
            plc_dtype = dtype_to_pylibcudf_type(dtype)
        return ColumnBase.from_pylibcudf(
            plc.Column.from_scalar(
                plc.Scalar.from_py(None, plc_dtype),
                row_count,
            )
        )._with_type_metadata(dtype)


def build_column(
    plc_column: plc.Column,
    dtype: DtypeObj,
    *,
    size: int,
    offset: int,
    null_count: int,
    exposed: bool,
) -> ColumnBase:
    """
    Build a Column of the appropriate type from the given parameters

    Parameters
    ----------
    plc_column : plc.Column
        The backing pylibcudf.Column
    dtype
        The dtype associated with the Column to construct
    size : int, optional
        The size of the column.
    offset : int, optional
        The offset of the column.
    null_count : int, optional
        The number of nulls in the column.
    exposed : bool, optional
        Whether the data buffers of the plc_column are exposed.
    """
    if isinstance(dtype, CategoricalDtype):
        return cudf.core.column.CategoricalColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, pd.DatetimeTZDtype):
        return cudf.core.column.datetime.DatetimeTZColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif dtype.kind == "M":
        return cudf.core.column.DatetimeColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif dtype.kind == "m":
        return cudf.core.column.TimeDeltaColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif (
        dtype == CUDF_STRING_DTYPE
        or dtype.kind == "U"
        or isinstance(dtype, pd.StringDtype)
        or (isinstance(dtype, pd.ArrowDtype) and dtype.kind == "U")
    ):
        return cudf.core.column.StringColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, ListDtype):
        return cudf.core.column.ListColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, IntervalDtype):
        return cudf.core.column.IntervalColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, StructDtype):
        return cudf.core.column.StructColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, cudf.Decimal64Dtype):
        return cudf.core.column.Decimal64Column(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, cudf.Decimal32Dtype):
        return cudf.core.column.Decimal32Column(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif isinstance(dtype, cudf.Decimal128Dtype):
        return cudf.core.column.Decimal128Column(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    elif dtype.kind in "iufb":
        return cudf.core.column.NumericalColumn(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
    else:
        raise TypeError(f"Unrecognized dtype: {dtype}")


def check_invalid_array(shape: tuple, dtype: np.dtype) -> None:
    """Invalid ndarrays properties that are not supported"""
    if len(shape) > 1:
        raise ValueError("Data must be 1-dimensional")
    elif dtype == "float16":
        raise TypeError("Unsupported type float16")


def maybe_reshape(
    arbitrary: Any,
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None,
    dtype: np.dtype,
) -> Any:
    """Reshape ndarrays compatible with cuDF columns."""
    if len(shape) == 0:
        arbitrary = cp.asarray(arbitrary)[np.newaxis]
    if not plc.column.is_c_contiguous(shape, strides, dtype.itemsize):
        arbitrary = cp.ascontiguousarray(arbitrary)
    return arbitrary


def as_memoryview(arbitrary: Any) -> memoryview | None:
    try:
        return memoryview(arbitrary)
    except TypeError:
        return None


def as_column(
    arbitrary: Any,
    nan_as_null: bool | None = None,
    dtype: Dtype | None = None,
    length: int | None = None,
) -> ColumnBase:
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
    # Always convert dtype up front so that downstream calls can assume it is a dtype
    # object rather than a string.
    if dtype is not None:
        dtype = cudf.dtype(dtype)

    if isinstance(arbitrary, (range, pd.RangeIndex, cudf.RangeIndex)):
        with acquire_spill_lock():
            column = ColumnBase.from_pylibcudf(
                plc.filling.sequence(
                    len(arbitrary),
                    pa_scalar_to_plc_scalar(
                        pa.scalar(arbitrary.start, type=pa.int64())
                    ),
                    pa_scalar_to_plc_scalar(
                        pa.scalar(arbitrary.step, type=pa.int64())
                    ),
                )
            )
        if cudf.get_option("default_integer_bitwidth") and dtype is None:
            dtype = np.dtype(
                f"i{cudf.get_option('default_integer_bitwidth') // 8}"
            )
        if dtype is not None:
            return column.astype(dtype)
        return column
    elif isinstance(arbitrary, (ColumnBase, cudf.Series, cudf.Index)):
        # Ignoring nan_as_null per the docstring
        if isinstance(arbitrary, cudf.Series):
            arbitrary = arbitrary._column
        elif isinstance(arbitrary, cudf.Index):
            arbitrary = arbitrary._column
        if dtype is not None:
            return arbitrary.astype(dtype)
        return arbitrary
    elif hasattr(arbitrary, "__cuda_array_interface__"):
        column = ColumnBase.from_cuda_array_interface(arbitrary)
        if nan_as_null is not False:
            column = column.nans_to_nulls()
        if dtype is not None:
            column = column.astype(dtype)
        return column
    elif isinstance(arbitrary, (pa.Array, pa.ChunkedArray)):
        column = ColumnBase.from_arrow(arbitrary)
        if nan_as_null is not False:
            column = column.nans_to_nulls()
        if dtype is not None:
            column = column.astype(dtype)
        return column
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
        elif isinstance(arbitrary.dtype, pd.IntervalDtype) and isinstance(
            arbitrary.dtype.subtype, pd.DatetimeTZDtype
        ):
            raise NotImplementedError(
                "cuDF does not yet support Intervals with timezone-aware datetimes"
            )
        elif isinstance(
            arbitrary.dtype,
            (pd.CategoricalDtype, pd.IntervalDtype, pd.DatetimeTZDtype),
        ):
            if isinstance(arbitrary.dtype, pd.DatetimeTZDtype):
                new_tz = get_compatible_timezone(arbitrary.dtype)
                arbitrary = arbitrary.astype(new_tz)
            if isinstance(arbitrary.dtype, pd.CategoricalDtype):
                if isinstance(
                    arbitrary.dtype.categories.dtype, pd.DatetimeTZDtype
                ):
                    new_tz = get_compatible_timezone(
                        arbitrary.dtype.categories.dtype
                    )
                    new_cats = arbitrary.dtype.categories.astype(new_tz)
                    new_dtype = pd.CategoricalDtype(
                        categories=new_cats, ordered=arbitrary.dtype.ordered
                    )
                    arbitrary = arbitrary.astype(new_dtype)
                elif (
                    isinstance(
                        arbitrary.dtype.categories.dtype, pd.IntervalDtype
                    )
                    and dtype is None
                ):
                    # Conversion to arrow converts IntervalDtype to StructDtype
                    dtype = CategoricalDtype(
                        categories=arbitrary.dtype.categories,
                        ordered=arbitrary.dtype.ordered,
                    )
            result = as_column(
                pa.array(arbitrary, from_pandas=True),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
            if (
                cudf.get_option("mode.pandas_compatible")
                and isinstance(arbitrary.dtype, pd.CategoricalDtype)
                and is_pandas_nullable_extension_dtype(
                    arbitrary.dtype.categories.dtype
                )
                and dtype is None
            ):
                # Store pandas extension dtype directly in the column's dtype property
                # TODO: Move this to near isinstance(arbitrary.dtype.categories.dtype, pd.IntervalDtype)
                # check above, for which merge should be working fully with pandas nullable extension dtypes.
                result = result._with_type_metadata(
                    CategoricalDtype(
                        categories=arbitrary.dtype.categories,
                        ordered=arbitrary.dtype.ordered,
                    )
                )
            return result
        elif is_pandas_nullable_extension_dtype(arbitrary.dtype):
            if (
                isinstance(arbitrary.dtype, pd.ArrowDtype)
                and (arrow_type := arbitrary.dtype.pyarrow_dtype) is not None
                and (
                    pa.types.is_date32(arrow_type)
                    or pa.types.is_binary(arrow_type)
                    or pa.types.is_dictionary(arrow_type)
                )
            ):
                raise NotImplementedError(
                    f"cuDF does not yet support {arbitrary.dtype}"
                )
            if isinstance(arbitrary, (pd.Series, pd.Index)):
                # pandas arrays define __arrow_array__ for better
                # pyarrow.array conversion
                arbitrary = arbitrary.array
            result = as_column(
                pa.array(arbitrary, from_pandas=True),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
            if cudf.get_option("mode.pandas_compatible"):
                # Store pandas extension dtype directly in the column's dtype property
                result = result._with_type_metadata(arbitrary.dtype)
            return result
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
                arbitrary = cp.asarray(arbitrary)
            return as_column(
                arbitrary, nan_as_null=nan_as_null, dtype=dtype, length=length
            )
        elif arbitrary.dtype.kind == "O":
            pyarrow_array = None
            if isinstance(arbitrary, NumpyExtensionArray):
                # infer_dtype does not handle NumpyExtensionArray
                arbitrary = np.array(arbitrary, dtype=object)
            inferred_dtype = infer_dtype(
                arbitrary,
                skipna=(
                    not cudf.get_option("mode.pandas_compatible")
                    and nan_as_null is not False
                ),
            )
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
            elif inferred_dtype == "boolean":
                if cudf.get_option("mode.pandas_compatible"):
                    if dtype != np.dtype("bool") or pd.isna(arbitrary).any():
                        raise MixedTypeError(
                            f"Cannot have mixed values with {inferred_dtype}"
                        )
                elif nan_as_null is False:
                    raise MixedTypeError(
                        f"Cannot have a {inferred_dtype} type with dtype=object"
                    )
            elif nan_as_null is False and inferred_dtype not in (
                "decimal",
                "empty",
                "string",
            ):
                if inferred_dtype == "floating":
                    raise MixedTypeError(
                        f"Cannot have a {inferred_dtype} type with dtype=object"
                    )
                try:
                    pyarrow_array = pa.array(arbitrary, from_pandas=False)
                except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
                    # Decimal can hold float("nan")
                    # All np.nan is not restricted by type
                    raise MixedTypeError(
                        f"Cannot have NaN with {inferred_dtype}"
                    )

            if pyarrow_array is None:
                pyarrow_array = pa.array(
                    arbitrary,
                    from_pandas=True,
                )
            if (
                cudf.get_option("mode.pandas_compatible")
                and inferred_dtype == "mixed"
                and not (
                    pa.types.is_list(pyarrow_array.type)
                    or pa.types.is_struct(pyarrow_array.type)
                    or pa.types.is_string(pyarrow_array.type)
                )
            ):
                raise MixedTypeError("Cannot create column with mixed types")
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

        pa_type = None
        if isinstance(arbitrary, pd.Interval) or _is_categorical_dtype(dtype):
            return as_column(
                pd.Series([arbitrary] * length),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
        elif (
            nan_as_null is True
            and isinstance(arbitrary, (np.floating, float))
            and np.isnan(arbitrary)
        ):
            if dtype is None:
                dtype = getattr(arbitrary, "dtype", np.dtype(np.float64))
            arbitrary = None
            pa_type = cudf_dtype_to_pa_type(dtype)
            dtype = None
        elif arbitrary is pd.NA or arbitrary is None:
            arbitrary = None
            if dtype is not None:
                pa_type = cudf_dtype_to_pa_type(dtype)
                if not cudf.get_option("mode.pandas_compatible"):
                    dtype = None
            else:
                raise ValueError(
                    "Need to pass dtype when passing pd.NA or None"
                )
        elif (
            isinstance(arbitrary, (pd.Timestamp, pd.Timedelta))
            or arbitrary is pd.NaT
        ):
            arbitrary = arbitrary.to_numpy()
        elif isinstance(arbitrary, (np.datetime64, np.timedelta64)):
            unit = np.datetime_data(arbitrary.dtype)[0]
            if unit not in {"s", "ms", "us", "ns"}:
                arbitrary = arbitrary.astype(
                    np.dtype(f"{arbitrary.dtype.kind}8[s]")
                )

        pa_scalar = pa.scalar(arbitrary, type=pa_type)
        if length == 0:
            if dtype is None:
                dtype = cudf_dtype_from_pa_type(pa_scalar.type)
            return column_empty(length, dtype=dtype)
        else:
            col = ColumnBase.from_pylibcudf(
                plc.Column.from_scalar(
                    pa_scalar_to_plc_scalar(pa_scalar), length
                )
            )
            if dtype is not None:
                col = col.astype(dtype)
            return col
    elif hasattr(arbitrary, "__array_interface__"):
        desc = arbitrary.__array_interface__
        check_invalid_array(desc["shape"], np.dtype(desc["typestr"]))

        # CUDF assumes values are always contiguous
        arbitrary = np.asarray(arbitrary, order="C")

        if arbitrary.ndim == 0:
            # TODO: Or treat as scalar?
            arbitrary = arbitrary[np.newaxis]

        if arbitrary.dtype.kind == "O":
            is_na = pd.isna(arbitrary)
            if is_na.any():
                if is_na.all():
                    # Avoid pyarrow converting np.ndarray[object] of all NaNs to float
                    raise MixedTypeError(
                        "Cannot have all NaN values with object dtype."
                    )
                arbitrary = pa.array(arbitrary)
            else:
                # Let pandas potentially infer object type
                # e.g. np.array([pd.Timestamp(...)], dtype=object) -> datetime64
                arbitrary = pd.Series(arbitrary)
            return as_column(arbitrary, dtype=dtype, nan_as_null=nan_as_null)
        elif arbitrary.dtype.kind in "SU":
            result_column = ColumnBase.from_arrow(pa.array(arbitrary))
            if dtype is not None:
                result_column = result_column.astype(dtype)
            return result_column
        elif arbitrary.dtype.kind in "biuf":
            if not arbitrary.dtype.isnative:
                # Not supported by pylibcudf
                arbitrary = arbitrary.astype(arbitrary.dtype.newbyteorder("="))
            result_column = ColumnBase.from_pylibcudf(
                plc.Column.from_array_interface(arbitrary)
            )
            if nan_as_null is not False:
                result_column = result_column.nans_to_nulls()
            if dtype is not None:
                result_column = result_column.astype(dtype)
            return result_column
        elif arbitrary.dtype.kind in "mM":
            time_unit = np.datetime_data(arbitrary.dtype)[0]
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
                if nan_as_null is not False:
                    # Convert NaT to NA, which pyarrow does by default
                    return as_column(
                        pa.array(arbitrary),
                        dtype=dtype,
                        nan_as_null=nan_as_null,
                    )
                # Consider NaT as NA in the mask
                # but maintain NaT as a value
                mask = plc.Column.from_array_interface(~is_nat)
            plc_column = plc.Column.from_array_interface(arbitrary)
            if mask is not None:
                plc_column = plc_column.with_mask(
                    *plc.transform.bools_to_mask(mask)
                )
            result_column = ColumnBase.from_pylibcudf(plc_column)
            if dtype is not None:
                result_column = result_column.astype(dtype)
            return result_column
        else:
            raise NotImplementedError(f"{arbitrary.dtype} not supported")
    elif (view := as_memoryview(arbitrary)) is not None:
        return as_column(
            np.asarray(view), dtype=dtype, nan_as_null=nan_as_null
        )
    elif not isinstance(arbitrary, (Iterable, Sequence)):
        raise TypeError(
            f"{type(arbitrary).__name__} must be an iterable or sequence."
        )
    elif isinstance(arbitrary, Iterator):
        arbitrary = list(arbitrary)

    # Start of arbitrary that's not handed above but dtype provided
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise NotImplementedError(
            "Use `tz_localize()` to construct timezone aware data."
        )
    elif isinstance(dtype, DecimalDtype):
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
        if isinstance(dtype, cudf.Decimal128Dtype):
            return cudf.core.column.Decimal128Column.from_arrow(data)
        elif isinstance(dtype, cudf.Decimal64Dtype):
            return cudf.core.column.Decimal64Column.from_arrow(data)
        elif isinstance(dtype, cudf.Decimal32Dtype):
            return cudf.core.column.Decimal32Column.from_arrow(data)
        else:
            raise NotImplementedError(f"{dtype} not implemented")
    elif isinstance(
        dtype,
        (
            pd.CategoricalDtype,
            CategoricalDtype,
            pd.IntervalDtype,
            IntervalDtype,
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
        if isinstance(dtype, (CategoricalDtype, IntervalDtype)):
            dtype = dtype.to_pandas()
            if isinstance(dtype, pd.IntervalDtype):
                # pd.Series(arbitrary) might be already inferred as IntervalDtype
                ser = pd.Series(arbitrary).astype(dtype)
            else:
                ser = pd.Series(arbitrary, dtype=dtype)
        elif dtype == object and not cudf.get_option("mode.pandas_compatible"):
            # Unlike pandas, interpret object as "str" instead of "python object"
            ser = pd.Series(arbitrary, dtype="str")
        else:
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

    from_pandas = nan_as_null is None or nan_as_null
    if dtype is not None:
        dtype = cudf.dtype(dtype)
        try:
            arbitrary = pa.array(
                arbitrary,
                type=cudf_dtype_to_pa_type(dtype),
                from_pandas=from_pandas,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError) as err:
            if err.args[0].startswith("Could not convert <NA>"):
                # nan_as_null=False, but we want to allow pd.NA values
                arbitrary = pa.array(
                    arbitrary,
                    type=cudf_dtype_to_pa_type(dtype),
                    from_pandas=True,
                )
            else:
                if isinstance(
                    dtype,
                    (
                        CategoricalDtype,
                        DecimalDtype,
                        IntervalDtype,
                        ListDtype,
                        StructDtype,
                    ),
                ):
                    dtype = dtype.to_pandas()
                arbitrary = pd.Series(arbitrary, dtype=dtype)
        return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)
    else:
        for element in arbitrary:
            # Carve-outs that cannot be parsed by pyarrow/pandas
            if is_column_like(element):
                # e.g. test_nested_series_from_sequence_data
                return cudf.core.column.ListColumn.from_sequences(arbitrary)
            elif isinstance(element, cp.ndarray):
                # e.g. test_series_from_cupy_scalars
                return as_column(
                    cp.array(arbitrary),
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
                # TODO: Need to re-visit this cast and fill_null
                # calls while addressing the following issue:
                # https://github.com/rapidsai/cudf/issues/14149
                arbitrary = arbitrary.cast(pa.float64())
                arbitrary = pc.fill_null(arbitrary, np.nan)
            if (
                cudf.get_option("default_integer_bitwidth")
                and pa.types.is_integer(arbitrary.type)
            ) or (
                cudf.get_option("default_float_bitwidth")
                and pa.types.is_floating(arbitrary.type)
            ):
                dtype = _maybe_convert_to_default_type(
                    np.dtype(arbitrary.type.to_pandas_dtype())
                )
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            arbitrary = pd.Series(arbitrary)
            if (
                cudf.get_option("default_integer_bitwidth")
                and arbitrary.dtype.kind in set("iu")
            ) or (
                cudf.get_option("default_float_bitwidth")
                and arbitrary.dtype.kind == "f"
            ):
                dtype = _maybe_convert_to_default_type(arbitrary.dtype)
        return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)


def serialize_columns(columns: list[ColumnBase]) -> tuple[list[dict], list]:
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
    headers: list[dict[Any, Any]] = []
    frames = []

    if len(columns) > 0:
        header_columns: list[tuple[dict, list]] = [
            c.device_serialize() for c in columns
        ]
        headers, column_frames = zip(*header_columns, strict=True)
        for f in column_frames:
            frames.extend(f)

    return headers, frames


def deserialize_columns(headers: list[dict], frames: list) -> list[ColumnBase]:
    """
    Construct a list of Columns from a list of headers
    and frames.
    """
    columns = []

    for meta in headers:
        col_frame_count = meta["frame_count"]
        col_typ = Serializable._name_type_map[meta["type-serialized-name"]]
        colobj = col_typ.deserialize(meta, frames[:col_frame_count])
        columns.append(colobj)
        # Advance frames
        frames = frames[col_frame_count:]

    return columns


def concat_columns(objs: Sequence[ColumnBase]) -> ColumnBase:
    """Concatenate a sequence of columns."""
    if len(objs) == 0:
        return column_empty(0, dtype=np.dtype(np.float64))

    # If all columns are `NumericalColumn` with different dtypes,
    # we cast them to a common dtype.
    # Notice, we can always cast pure null columns
    not_null_col_dtypes = [o.dtype for o in objs if o.null_count != len(o)]
    if len(not_null_col_dtypes) and all(
        is_dtype_obj_numeric(dtype, include_decimal=False)
        and dtype.kind == "M"
        for dtype in not_null_col_dtypes
    ):
        common_dtype = find_common_type(not_null_col_dtypes)
        # Cast all columns to the common dtype
        objs = [obj.astype(common_dtype) for obj in objs]

    # Find the first non-null column:
    head = next((obj for obj in objs if obj.null_count != len(obj)), objs[0])

    replacement_cols: dict[int, ColumnBase] = {}
    for i, obj in enumerate(objs):
        # Check that all columns are the same type:
        if not is_dtype_equal(obj.dtype, head.dtype):
            # if all null, cast to appropriate dtype
            if obj.null_count == len(obj):
                replacement_cols[i] = column_empty(
                    row_count=len(obj), dtype=head.dtype
                )
            else:
                raise ValueError("All columns must be the same type")
    if replacement_cols:
        if len(replacement_cols) == len(objs):
            new_objs: Sequence[ColumnBase] = list(replacement_cols.values())
        else:
            new_objs = list(objs)
            for idx, col in replacement_cols.items():
                new_objs[idx] = col
    else:
        new_objs = objs

    # TODO: This logic should be generalized to a dispatch to
    # ColumnBase._concat so that all subclasses can override necessary
    # behavior. However, at the moment it's not clear what that API should look
    # like, so CategoricalColumn simply implements a minimal working API.
    if all(isinstance(o.dtype, CategoricalDtype) for o in new_objs):
        return cudf.core.column.categorical.CategoricalColumn._concat(
            cast(
                MutableSequence[
                    cudf.core.column.categorical.CategoricalColumn
                ],
                new_objs,
            )
        )

    newsize = sum(map(len, new_objs))
    if newsize > np.iinfo(SIZE_TYPE_DTYPE).max:
        raise MemoryError(
            f"Result of concat cannot have size > {SIZE_TYPE_DTYPE}_MAX"
        )
    elif newsize == 0:
        return column_empty(0, head.dtype)

    # Filter out inputs that have 0 length, then concatenate.
    objs_with_len = [o for o in new_objs if len(o)]
    with acquire_spill_lock():
        return ColumnBase.from_pylibcudf(
            plc.concatenate.concatenate(
                [col.to_pylibcudf(mode="read") for col in objs_with_len]
            )
        )._with_type_metadata(objs_with_len[0].dtype)
