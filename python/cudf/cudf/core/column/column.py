# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import warnings
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    MutableSequence,
    Sequence,
)
from contextlib import ExitStack
from decimal import Decimal
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

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
)
from cudf.core._internals.timezones import get_compatible_timezone
from cudf.core.abc import Serializable
from cudf.core.buffer import (
    Buffer,
    as_buffer,
)
from cudf.core.column.utils import access_columns
from cudf.core.copy_types import GatherMap
from cudf.core.dtype.validators import (
    is_dtype_obj_decimal,
    is_dtype_obj_interval,
    is_dtype_obj_list,
    is_dtype_obj_numeric,
    is_dtype_obj_struct,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    DecimalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
    _dtype_to_metadata,
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
    get_dtype_of_same_kind,
    is_column_like,
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
    from types import TracebackType

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


class MaskCAIWrapper:
    # A wrapper that exposes the __cuda_array_interface__ of a mask that accounts for
    # the mask being a bitmask in the mask size calculation.

    def __init__(self, mask: Any) -> None:
        self._mask = mask

    @property
    def __cuda_array_interface__(self) -> Mapping:
        cai = self._mask.__cuda_array_interface__.copy()
        cai["shape"] = (
            plc.null_mask.bitmask_allocation_size_bytes(cai["shape"][0]),
        )
        return cai

    @property
    def ptr(self) -> int:
        """Device pointer (Span protocol)."""
        return self._mask.__cuda_array_interface__["data"][0]

    @property
    def size(self) -> int:
        """Size in bytes (Span protocol)."""
        return plc.null_mask.bitmask_allocation_size_bytes(
            self._mask.__cuda_array_interface__["shape"][0]
        )


def _handle_nulls(arrow_array: pa.Array) -> pa.Array:
    # Recursively replace all-null nested arrays with arrow NullArrays and make all
    # types nullable in the schema even if the columns contain no nulls
    array_type = arrow_array.type

    # Empty nested or string types should become pyarrow null arrays instead of the
    # normal array classes to match how pyarrow ingests such pandas objects.
    if (
        pa.types.is_nested(array_type)
        or pa.types.is_string(array_type)
        or pa.types.is_large_string(array_type)
    ) and (arrow_array.null_count == len(arrow_array)):
        return pa.NullArray.from_buffers(
            pa.null(), len(arrow_array), [pa.py_buffer(b"")]
        )

    if pa.types.is_struct(array_type):
        arrow_array = cast("pa.StructArray", arrow_array)

        new_fields = []
        requires_reconstruction = False
        for i, subfield in enumerate(array_type):
            new_field_array = field_array = arrow_array.field(i)
            field_type = subfield.type
            if (
                pa.types.is_nested(field_type)
                or pa.types.is_string(field_type)
                or pa.types.is_large_string(field_type)
            ):
                new_field_array = _handle_nulls(field_array)
            new_fields.append(new_field_array)
            # Reconstruct if we replaced nulls in children or need nullability change
            requires_reconstruction = (
                requires_reconstruction
                or (new_field_array is not field_array)
                or not subfield.nullable
                and not pa.types.is_null(field_type)
            )

        if requires_reconstruction:
            new_struct_type = pa.struct(
                [
                    pa.field(at.name, nf.type, nullable=True)
                    for at, nf in zip(array_type, new_fields, strict=True)
                ]
            )
            # Only need validity buffer for structs
            buffers = cast("list[pa.Buffer]", arrow_array.buffers()[:1])
            return pa.StructArray.from_buffers(
                new_struct_type,
                len(arrow_array),
                buffers,
                children=new_fields,
                null_count=arrow_array.null_count,
            )
    elif pa.types.is_list(array_type):
        arrow_array = cast("pa.ListArray", arrow_array)

        values = arrow_array.values
        new_values = _handle_nulls(values)

        value_field = array_type.value_field
        has_non_nullable_field = (
            not value_field.nullable and not pa.types.is_null(value_field.type)
        )

        if new_values is not values or has_non_nullable_field:
            buffers = cast("list[pa.Buffer]", arrow_array.buffers()[:2])
            list_type = pa.list_(
                pa.field(value_field.name, new_values.type, nullable=True)
            )
            return pa.ListArray.from_buffers(
                list_type,
                len(arrow_array),
                buffers,
                children=[new_values],
                null_count=arrow_array.null_count,
            )
    # For other primitives (int/float/etc), preserve type even if all null
    return arrow_array


class _ColumnAccessContext:
    """Context manager for access mode control on underlying buffers."""

    __slots__ = ("_column", "_kwargs", "_stack")

    _stack: ExitStack

    def __init__(self, column: ColumnBase, **kwargs: Any):
        """Initialize column access context.

        Parameters
        ----------
        column : ColumnBase
            The column to manage access for.
        **kwargs
            Parameters to propagate to buffer access (e.g., mode, scope).
        """
        self._column = column
        self._kwargs = kwargs
        self._stack = ExitStack()

    def _enter_plc(self, plc_column: plc.Column) -> None:
        """Enter context for all child columns."""
        if (data := plc_column.data()) is not None:
            self._stack.enter_context(
                cast("Buffer", data).access(**self._kwargs)
            )
        if (mask := plc_column.null_mask()) is not None:
            self._stack.enter_context(
                cast("Buffer", mask).access(**self._kwargs)
            )
        for child in plc_column.children():
            self._enter_plc(child)

    def __enter__(self) -> ColumnBase:
        """Enter the context, setting up access for all column buffers."""
        # Propagate all kwargs transparently to all buffers
        self._enter_plc(self._column.plc_column)
        return self._column

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self._stack.__exit__(exc_type, exc_val, exc_tb)
        return False


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
        "min",
        "max",
    }
    _VALID_PLC_TYPES: ClassVar[set[plc.TypeId]] = set()
    plc_column: plc.Column
    _dtype: DtypeObj
    _distinct_count: dict[bool, int]
    _exposed_buffers: set[Buffer]
    _CACHED_PROPERTY_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Pre-compute the set of all cached properties for efficient cache clearing
        cached_props = set()
        for base_cls in cls.__mro__:
            for attr_name, attr_value in base_cls.__dict__.items():
                if isinstance(attr_value, cached_property):
                    cached_props.add(attr_name)
        cls._CACHED_PROPERTY_NAMES = frozenset(cached_props)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "ColumnBase and its subclasses must be instantiated via from_pylibcudf."
        )

    @classmethod
    def _validate_args(
        cls, plc_column: plc.Column, dtype: DtypeObj
    ) -> tuple[plc.Column, DtypeObj]:
        """Validate and return plc_column and dtype arguments for column construction."""
        if not (
            isinstance(plc_column, plc.Column)
            and plc_column.type().id() in cls._VALID_PLC_TYPES
        ):
            raise ValueError(
                f"plc_column must be a pylibcudf.Column with a TypeId in {cls._VALID_PLC_TYPES}"
            )
        return plc_column, dtype

    @property
    def _PANDAS_NA_VALUE(self) -> ScalarLike:
        """Return appropriate NA value based on dtype."""
        if cudf.get_option(
            "mode.pandas_compatible"
        ) and is_pandas_nullable_extension_dtype(self.dtype):
            return self.dtype.na_value
        return pd.NA

    @property
    def dtype(self) -> DtypeObj:
        return self._dtype

    @property
    def size(self) -> int:
        return self.plc_column.size()

    @property
    def data(self) -> None | Buffer:
        """Get data buffer from pylibcudf column."""
        return cast("Buffer | None", self.plc_column.data())

    @property
    def nullable(self) -> bool:
        return self.mask is not None

    def has_nulls(self, include_nan: bool = False) -> bool:
        """Check if column has null values.

        NaN inclusion is supported for specific dtypes only.
        """
        return int(self.null_count) != 0

    @property
    def mask(self) -> None | Buffer:
        """Get mask buffer from pylibcudf column."""
        return cast("Buffer | None", self.plc_column.null_mask())

    def access(self, **kwargs: Any) -> _ColumnAccessContext:
        """Context manager for controlled buffer access.

        Mediates access to all the underlying buffers of the column. Within this
        context, all their ptr accesses will respect the specified access parameters.

        Applies all parameters transparently to all underlying buffers in the
        column hierarchy (data, mask, and children).

        Parameters
        ----------
        **kwargs
            Parameters for buffer access (e.g., mode, scope).
            - mode : {"read", "write"} - Access mode for copy-on-write
            - scope : {"internal", "external"} - Spill scope (SpillableBuffer only)

        Returns
        -------
        _ColumnAccessContext
            A context manager that manages access to all column buffers.
        """
        return _ColumnAccessContext(self, **kwargs)

    def _clear_cache(self) -> None:
        self._distinct_count.clear()
        for attr_name in self._CACHED_PROPERTY_NAMES:
            try:
                delattr(self, attr_name)
            except AttributeError:
                pass

    def set_mask(self, mask: Buffer | None, null_count: int) -> Self:
        """
        Replaces the mask buffer of the column and returns a new column.

        Parameters
        ----------
        mask : Buffer or None
            The new null mask buffer, or None to clear the mask.
        null_count : int
            The number of null values.
        """
        if isinstance(mask, Buffer):
            new_mask = mask
            new_null_count = null_count
        elif mask is None:
            new_mask = None
            new_null_count = 0
        else:
            raise ValueError(
                f"Expected a Buffer object or None for mask, got {type(mask).__name__}"
            )
        new_plc_column = self.plc_column.with_mask(new_mask, new_null_count)
        return cast(
            "Self",
            ColumnBase.create(new_plc_column, self.dtype),
        )

    @property
    def null_count(self) -> int:
        return self.plc_column.null_count()

    @property
    def offset(self) -> int:
        return self.plc_column.offset()

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
            self._dtype = other_col._dtype
            self.plc_column = other_col.plc_column
            self._clear_cache()
            return None
        else:
            return other_col

    @staticmethod
    def _unwrap_buffer_to_span(buffer: Buffer | Any | None) -> Any:
        """Unwrap a cudf Buffer to a span.

        If the input is a cudf Buffer, extract the underlying span. Otherwise,
        return the input as-is (already a span or None).

        Parameters
        ----------
        buffer : Buffer or None
            The buffer to unwrap.

        Returns
        -------
        plc.span.Span or None
            The underlying span, or None if input was None.
        """
        if buffer is None:
            return None
        return cast("plc.span.Span", cast("Buffer", buffer).owner.owner)

    @staticmethod
    def _unwrap_buffers(plc_column: plc.Column) -> plc.Column:
        """Recursively unwrap all buffers in a pylibcudf.Column.

        This function will traverse the provided pylibcudf Column and unwrap
        all data and mask buffers from cudf Buffers, removing cudf memory
        semantics.

        Parameters
        ----------
        col : pylibcudf.Column
            The column to unwrap.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column with unwrapped buffers.
        """
        data = ColumnBase._unwrap_buffer_to_span(plc_column.data())
        mask = ColumnBase._unwrap_buffer_to_span(plc_column.null_mask())

        unwrapped_children = [
            ColumnBase._unwrap_buffers(child)
            for child in plc_column.children()
        ]

        return plc.Column(
            data_type=plc_column.type(),
            size=plc_column.size(),
            data=data,
            mask=mask,
            null_count=plc_column.null_count(),
            offset=plc_column.offset(),
            children=unwrapped_children,
            validate=False,
        )

    def to_pylibcudf(self) -> plc.Column:
        """Convert this Column to a pylibcudf.Column.

        This function will generate a pylibcudf Column with unwrapped buffers,
        removing cudf memory semantics. The result is a view of the existing data.
        It is always a zero-copy operation, so changes to the pylibcudf data will
        be reflected back to the cudf object and vice versa. Users are responsible
        for making a copy if they wish to avoid this behavior. Note that due to
        copy-on-write semantics in cudf modifying the pylibcudf object may affect
        multiple preexisting cudf objects viewing the same buffer.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column with unwrapped buffers.
        """
        return ColumnBase._unwrap_buffers(self.plc_column)

    @staticmethod
    def _wrap_buffer_or_span(
        buffer_or_span: Buffer | Any | None,
    ) -> Buffer | None:
        """Wrap a buffer or span in a cudf Buffer.

        If the input is already a Buffer, it will be shallow-copied to track
        that we are now sharing the same BufferOwner. If it's a gpumemoryview
        or other span, it will be wrapped in a new Buffer.

        Parameters
        ----------
        buffer_or_span : Buffer, gpumemoryview, or None
            The buffer or span to wrap.

        Returns
        -------
        Buffer or None
            A wrapped Buffer, or None if input was None.
        """
        if buffer_or_span is None:
            return None
        if isinstance(buffer_or_span, Buffer):
            return buffer_or_span.copy(deep=False)
        return as_buffer(buffer_or_span)

    @staticmethod
    def _wrap_buffers(col: plc.Column) -> plc.Column:
        """Recursively wrap all buffers in a pylibcudf.Column.

        This function will traverse the provided pylibcudf Column and wrap
        all data and mask buffers in cudf Buffers, ensuring that cudf memory
        semantics are preserved when using the resulting Column.

        Parameters
        ----------
        col : pylibcudf.Column
            The column to wrap.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column with wrapped buffers.
        """
        # Convert unsupported types to supported types
        # TODO: Removing this conversion causes some pandas tests to fail, but others
        # pass, so we need to investigate further to understand whether we should be
        # doing this conversion on a more per-algorithm basis or something.
        if col.type().id() == plc.TypeId.TIMESTAMP_DAYS:
            col = plc.unary.cast(
                col, plc.DataType(plc.TypeId.TIMESTAMP_SECONDS)
            )
        elif col.type().id() == plc.TypeId.EMPTY:
            new_dtype = plc.DataType(plc.TypeId.INT8)
            col = plc.column_factories.make_numeric_column(
                new_dtype, col.size(), plc.types.MaskState.ALL_NULL
            )

        data = ColumnBase._wrap_buffer_or_span(col.data())
        mask = ColumnBase._wrap_buffer_or_span(col.null_mask())

        wrapped_children = [
            ColumnBase._wrap_buffers(child) for child in col.children()
        ]

        return plc.Column(
            data_type=col.type(),
            size=col.size(),
            data=data,
            mask=mask,
            null_count=col.null_count(),
            offset=col.offset(),
            children=wrapped_children,
            validate=False,
        )

    @staticmethod
    def create(col: plc.Column, dtype: DtypeObj) -> ColumnBase:
        """
        Create a Column from a pylibcudf.Column with an explicit cudf dtype.

        This is the primary factory for ColumnBase construction. It always requires
        an explicit dtype to ensure type safety. If you need to infer the dtype from
        the pylibcudf Column, use dtype_from_pylibcudf_column() first:

            dtype = dtype_from_pylibcudf_column(plc_col)
            col = ColumnBase.create(plc_col, dtype)

        Note that the input col is never directly placed into the resulting ColumnBase.
        Rather, a new plc.Column is created with the exact same properties but with
        suitable shallow copies of the buffers wrapped in cudf Buffers to ensure
        consistent behavior with respect to memory semantics like copy-on-write.
        """
        # Wrap buffers recursively
        wrapped = ColumnBase._wrap_buffers(col)

        # Dispatch to the appropriate subclass based on dtype
        target_cls = ColumnBase._dispatch_subclass_from_dtype(dtype)

        # Validate dtype compatibility with the column structure using the
        # target subclass's _validate_args method (includes recursive validation)
        wrapped, dtype = target_cls._validate_args(wrapped, dtype)

        # Construct the instance using the subclass's _from_preprocessed method
        # Skip validation since we already validated above
        return target_cls._from_preprocessed(
            plc_column=wrapped,
            dtype=dtype,
            validate=False,
        )

    @staticmethod
    def _dispatch_subclass_from_dtype(dtype: DtypeObj) -> type[ColumnBase]:
        """
        Dispatch to the appropriate ColumnBase subclass based on dtype.

        This function determines which ColumnBase subclass should be used
        to construct a column with the given dtype.
        """
        # Special pandas extension types
        if isinstance(dtype, pd.DatetimeTZDtype):
            return cudf.core.column.datetime.DatetimeTZColumn
        if isinstance(dtype, CategoricalDtype):
            return cudf.core.column.CategoricalColumn

        # Temporal types (by kind)
        if dtype.kind == "M":
            return cudf.core.column.DatetimeColumn
        if dtype.kind == "m":
            return cudf.core.column.TimeDeltaColumn

        # String types
        if (
            dtype == CUDF_STRING_DTYPE
            or (hasattr(dtype, "kind") and dtype.kind == "U")
            or isinstance(dtype, pd.StringDtype)
            or (isinstance(dtype, pd.ArrowDtype) and dtype.kind == "U")
        ):
            return cudf.core.column.StringColumn

        # cuDF custom types
        if isinstance(dtype, ListDtype):
            return cudf.core.column.ListColumn
        if isinstance(dtype, IntervalDtype):
            return cudf.core.column.IntervalColumn
        if isinstance(dtype, StructDtype):
            return cudf.core.column.StructColumn

        # Decimal types
        if isinstance(dtype, cudf.Decimal128Dtype):
            return cudf.core.column.Decimal128Column
        if isinstance(dtype, cudf.Decimal64Dtype):
            return cudf.core.column.Decimal64Column
        if isinstance(dtype, cudf.Decimal32Dtype):
            return cudf.core.column.Decimal32Column

        # Numerical types
        if dtype.kind in "iufb":
            return cudf.core.column.NumericalColumn

        raise TypeError(f"Unrecognized dtype: {dtype}")

    @staticmethod
    def _validate_dtype_recursively(col: plc.Column, dtype: DtypeObj) -> None:
        """
        Validate dtype compatibility by dispatching to the appropriate ColumnBase
        subclass's _validate_args method.

        This method is used for recursive validation in nested types (List, Struct,
        Interval). It dispatches to the correct ColumnBase subclass based on dtype
        and calls its _validate_args method, which may recursively call this method
        for nested children.

        Parameters
        ----------
        col : plc.Column
            The pylibcudf Column to validate.
        dtype : DtypeObj
            The cudf dtype to validate against.

        Raises
        ------
        ValueError
            If the dtype is incompatible with the Column.
        """
        # Skip validation for empty columns (INT8 with all nulls). These are created
        # by _wrap_buffers() from EMPTY columns and may have inaccurate dtype metadata.
        # For example, an empty list [] has element_type=object but child is INT8.
        if (
            col.type().id() == plc.TypeId.INT8
            and col.null_count() == col.size()
        ):
            return

        # Dispatch to the appropriate subclass and use its _validate_args
        target_cls = ColumnBase._dispatch_subclass_from_dtype(dtype)
        target_cls._validate_args(col, dtype)

    @staticmethod
    def from_pylibcudf(col: plc.Column) -> ColumnBase:
        """Create a Column from a pylibcudf.Column.

        This function will generate a Column pointing to the provided pylibcudf
        Column. It will directly access the data and mask buffers of the
        pylibcudf Column, so the newly created object is not tied to the
        lifetime of the original pylibcudf.Column.

        Parameters
        ----------
        col : pylibcudf.Column
            The object to copy.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column referencing the same data.
        """
        # Wrap buffers first so that dtypes are compatible with dtype_from_pylibcudf_column
        wrapped = ColumnBase._wrap_buffers(col)
        dtype = dtype_from_pylibcudf_column(wrapped)
        return ColumnBase.create(wrapped, dtype)

    @classmethod
    def _from_preprocessed(
        cls,
        plc_column: plc.Column,
        dtype: DtypeObj,
        validate: bool = True,
    ) -> Self:
        # TODO: This function bypassess some of the buffer copying/wrapping that would
        # be done in from_pylibcudf, so it is only ever safe to call this in situations
        # where we know that the plc_column and children are already properly wrapped.
        # Ideally we should get rid of this altogether eventually and inline its logic
        # in from_pylibcudf, but for now it is necessary for the various
        # _with_type_metadata calls.
        self = cls.__new__(cls)
        if validate:
            plc_column, dtype = self._validate_args(plc_column, dtype)
        self.plc_column = plc_column
        self._dtype = dtype
        self._distinct_count = {}
        # The set of exposed buffers associated with this column. These buffers must be
        # kept alive for the lifetime of this column since anything that accessed the
        # CAI of this column will still be pointing to those buffers. As such objects
        # are destroyed, all references to this column will be removed as well,
        # triggering the destruction of the exposed buffers.
        self._exposed_buffers = set()
        return self

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
        # Reshape ndarrays compatible with cuDF columns
        if len(cai["shape"]) == 0:
            arbitrary = cp.asarray(arbitrary)[np.newaxis]
        if not plc.column.is_c_contiguous(
            cai["shape"], cai["strides"], cai_dtype.itemsize
        ):
            arbitrary = cp.ascontiguousarray(arbitrary)

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
        )
        if mask is not None:
            cai_mask = mask.__cuda_array_interface__
            if cai_mask["typestr"][1] == "t":
                mask_buff = as_buffer(MaskCAIWrapper(mask))
            elif cai_mask["typestr"][1] == "b":
                mask_buff, null_count = ColumnBase.from_cuda_array_interface(
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
            # Compute null_count for external masks that don't provide it
            if cai_mask["typestr"][1] != "b":
                null_count = plc.null_mask.null_count(
                    mask_buff, 0, column.size
                )
            column = column.set_mask(mask_buff, null_count)
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
            return self.astype(CUDF_STRING_DTYPE).fillna(
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
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        pa_array = self.to_arrow()

        if arrow_type or (
            not nullable and isinstance(self.dtype, pd.ArrowDtype)
        ):
            return pd.Index(
                pd.arrays.ArrowExtensionArray(pa_array), copy=False
            )
        elif nullable or (
            not arrow_type and is_pandas_nullable_extension_dtype(self.dtype)
        ):
            pandas_nullable_dtype = np_dtypes_to_pandas_dtypes.get(
                self.dtype, self.dtype
            )
            pandas_array = pandas_nullable_dtype.__from_arrow__(pa_array)
            return pd.Index(pandas_array, copy=False)
        else:
            # xref https://github.com/rapidsai/cudf/issues/21120
            # TODO: Revisit using pa_array.to_pandas() once pandas 3.0 is supported
            return pd.Index(
                pa_array.to_numpy(zero_copy_only=False, writable=True),
                copy=False,
            )

    @property
    def values_host(self) -> np.ndarray:
        """
        Return a numpy representation of the Column.
        """
        return self.to_pandas().to_numpy()

    @property
    def values(self) -> cp.ndarray:
        """Return a CuPy representation of the Column."""
        raise NotImplementedError(f"CuPy does not support {self.dtype}")

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        raise NotImplementedError

    def clip(self, lo: ScalarLike, hi: ScalarLike) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.replace.clamp(
                self.plc_column,
                pa_scalar_to_plc_scalar(
                    pa.scalar(lo, type=cudf_dtype_to_pa_type(self.dtype))
                ),
                pa_scalar_to_plc_scalar(
                    pa.scalar(hi, type=cudf_dtype_to_pa_type(self.dtype))
                ),
            )
            return cast(
                "Self",
                type(self).from_pylibcudf(plc_column),
            )

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
        return bool(ret.all())

    def all(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        if self.size == 0:
            return True
        if self.null_count == self.size:
            if not skipna:
                return _get_nan_for_dtype(self.dtype)
            else:
                return True

        # For all(), we want NaN values to be treated as truthy.
        # Call _reduce() with skipna=True to get the boolean result.
        result = self._reduce(
            "all", skipna=True, min_count=min_count, **kwargs
        )
        if np.isnan(result):
            # Empty after dropping NaN/nulls - return np.bool_
            result = np.bool_(True)

        # For pandas nullable extension dtypes with skipna=False and nulls, return NaN
        if (
            result
            and not skipna
            and self.null_count > 0
            and is_pandas_nullable_extension_dtype(self.dtype)
        ):
            return _get_nan_for_dtype(self.dtype)
        return result

    def any(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        if self.size == 0:
            return False
        if not skipna and (self.has_nulls() or self.nan_count > 0):
            return True
        elif skipna and self.null_count == self.size:
            return False

        # For any(), we want NaN values to be treated as truthy.
        # Call _reduce() with skipna=True to get the boolean result.
        result = self._reduce(
            "any", skipna=True, min_count=min_count, **kwargs
        )
        if np.isnan(result):
            # Empty after dropping NaN/nulls
            # If skipna=False, NaN values should be treated as truthy
            result = np.bool_(not skipna)
        return result

    def dropna(self) -> Self:
        if self.has_nulls():
            with self.access(mode="read", scope="internal"):
                plc_table = plc.stream_compaction.drop_nulls(
                    plc.Table([self.plc_column]),
                    [0],
                    1,
                )
                return cast(
                    "Self",
                    ColumnBase.create(plc_table.columns()[0], self.dtype),
                )
        else:
            return self.copy()

    def to_arrow(self) -> pa.Array:
        with self.access(mode="read", scope="internal"):
            return _handle_nulls(
                self.plc_column.to_arrow(
                    metadata=_dtype_to_metadata(self.dtype)
                )
            )

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
                dict_array = cast("pa.DictionaryArray", array)
                codes: pa.Array | pa.ChunkedArray = dict_array.indices
                dictionary: pa.Array | pa.ChunkedArray = dict_array.dictionary
            else:
                codes = pa.chunked_array(
                    [
                        cast("pa.DictionaryArray", chunk).indices
                        for chunk in array.chunks
                    ],
                    type=array.type.index_type,
                )
                dictionary = pc.unique(
                    pa.chunked_array(
                        [
                            cast("pa.DictionaryArray", chunk).dictionary
                            for chunk in array.chunks
                        ],
                        type=array.type.value_type,
                    )
                )
            result = cls.from_pylibcudf(plc.Column.from_arrow(codes))
            categories = cls.from_pylibcudf(plc.Column.from_arrow(dictionary))
            return result._with_type_metadata(
                CategoricalDtype(
                    categories=categories, ordered=array.type.ordered
                )
            )
        else:
            result = cls.from_pylibcudf(plc.Column.from_arrow(array))
            return result._with_type_metadata(
                cudf_dtype_from_pa_type(array.type)
            )

    def _get_mask_as_column(self) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.transform.mask_to_bools(
                self.mask.ptr,  # type: ignore[union-attr]
                self.offset,
                self.offset + len(self),
            )
            return type(self).from_pylibcudf(plc_column)

    @staticmethod
    def _plc_memory_usage(col: plc.Column) -> int:
        n = 0
        if (data := col.data()) is not None:
            try:
                # Only count the actual data in use, not the entire base buffer
                # For sliced columns, col.size() * itemsize gives actual usage
                typestr: str = col.type().typestr  # type: ignore[assignment]
                itemsize = np.dtype(typestr).itemsize
                n += col.size() * itemsize
            except NotImplementedError:
                # No typestr available (e.g. STRING)
                n += data.size
        if col.null_mask() is not None:
            n += plc.null_mask.bitmask_allocation_size_bytes(col.size())
        for child in col.children():
            n += ColumnBase._plc_memory_usage(child)
        return n

    @cached_property
    def memory_usage(self) -> int:
        return self._plc_memory_usage(self.plc_column)

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
            with self.access(mode="read", scope="internal"):
                result = cast(
                    "Self",
                    type(self).from_pylibcudf(
                        plc.filling.fill(
                            self.plc_column,
                            begin,
                            end,
                            fill_value,
                        )
                    ),
                )
            if self.dtype == CUDF_STRING_DTYPE:
                return self._mimic_inplace(result, inplace=True)
            return result

        if not fill_value.is_valid(DEFAULT_STREAM) and not self.nullable:
            # Create mask sized for base buffer to preserve view semantics
            mask = as_buffer(
                plc.null_mask.create_null_mask(
                    self.size, plc.types.MaskState.ALL_VALID
                )
            )
            self.plc_column = self.plc_column.with_mask(
                mask, 0, validate=False
            )
            self._clear_cache()

        with self.access(mode="read", scope="internal"):
            with self.access(mode="write"):
                plc.filling.fill_in_place(
                    self.plc_column,
                    begin,
                    end,
                    fill_value,
                )
            self._clear_cache()
        return self

    def shift(self, offset: int, fill_value: ScalarLike) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_fill_value = self._scalar_to_plc_scalar(fill_value)
            plc_col = plc.copying.shift(
                self.plc_column,
                offset,
                plc_fill_value,
            )
            return cast(
                "Self",
                type(self).from_pylibcudf(plc_col),
            )

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
        plc_col = self.plc_column
        if deep:
            plc_col = plc_col.copy()
        return cast("Self", ColumnBase.create(plc_col, self.dtype))

    def element_indexing(self, index: int) -> ScalarLike:
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
        with self.access(mode="read", scope="internal"):
            plc_scalar = plc.copying.get_element(
                self.plc_column,
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
        if stop < 0 and not (stride < 0 and stop == -1 and start >= 0):
            stop = stop + len(self)
        if start < 0:
            start = start + len(self)
        if (stride > 0 and start >= stop) or (stride < 0 and start <= stop):
            return cast("Self", column_empty(0, self.dtype))
        # compute mask slice
        if stride == 1:
            with self.access(mode="read", scope="internal"):
                (result,) = plc.copying.slice(
                    self.plc_column,
                    [start, stop],
                )
            return cast("Self", ColumnBase.create(result, self.dtype))
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
        self, other: ColumnBase | ScalarLike, bool_fill_value: bool
    ) -> ColumnBase:
        # Might be able to remove if we share more of
        # DatetimeColumn._binaryop & TimedeltaColumn._binaryop
        result_mask = None
        if self.has_nulls():
            if isinstance(other, ColumnBase) and other.has_nulls():
                result_mask = (
                    self._get_mask_as_column() & other._get_mask_as_column()
                )
            elif (
                not isinstance(other, ColumnBase)
                and other not in {np.nan, None, pd.NaT, float("nan")}
                and not (isinstance(other, Decimal) and other.is_nan())
                and not (isinstance(other, float) and np.isnan(other))
            ):
                result_mask = self._get_mask_as_column()
        elif isinstance(other, ColumnBase) and other.has_nulls():
            result_mask = other._get_mask_as_column()

        result_col = as_column(
            bool_fill_value,
            dtype=get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
            length=len(self),
        )
        if result_mask is not None:
            mask_buff, null_count = result_mask.as_mask()
            result_col = result_col.set_mask(mask_buff, null_count)
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
                with self.access(mode="read", scope="internal"):
                    with self.access(mode="write"):
                        return cast(
                            "Self",
                            type(self).from_pylibcudf(
                                plc.copying.copy_range(
                                    value.plc_column,
                                    self.plc_column,
                                    0,
                                    num_keys,
                                    start,
                                )
                            ),
                        )

        # step != 1, create a scatter map with arange
        scatter_map = cast(
            "cudf.core.column.NumericalColumn",
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
            with self.access(mode="read", scope="internal"):
                with self.access(mode="write"):
                    plc_table = plc.copying.boolean_mask_scatter(
                        plc.Table([value.plc_column])
                        if isinstance(value, ColumnBase)
                        else [value],
                        plc.Table([self.plc_column]),
                        key.plc_column,
                    )
                return cast(
                    "Self",
                    ColumnBase.create(plc_table.columns()[0], self.dtype),
                )
        else:
            return cast(
                "Self",
                ColumnBase.create(
                    copying.scatter(
                        cast("list[plc.Scalar]", [value])
                        if isinstance(value, plc.Scalar)
                        else cast("list[ColumnBase]", [value]),
                        key,
                        [self],
                        bounds_check=bounds_check,
                    )[0],
                    self.dtype,
                ),
            )

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

    def replace(
        self, values_to_replace: Self, replacement_values: Self
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                "Self",
                type(self).from_pylibcudf(
                    plc.replace.find_and_replace_all(
                        self.plc_column,
                        values_to_replace.plc_column,
                        replacement_values.plc_column,
                    )
                ),
            )

    def repeat(self, repeats: int) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                "Self",
                type(self).from_pylibcudf(
                    plc.filling.repeat(
                        plc.Table([self.plc_column]), repeats
                    ).columns()[0]
                ),
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

        with self.access(mode="read", scope="internal"):
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
                plc_replace = fill_value.plc_column
            plc_column = plc.replace.replace_nulls(
                input_col.plc_column,
                plc_replace,
            )
        return cast(
            "Self",
            ColumnBase.create(plc_column, self.dtype),
        )

    def is_valid(self) -> ColumnBase:
        """Identify non-null values"""
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.unary.is_valid(self.plc_column), np.dtype(np.bool_)
            )

    def isnan(self) -> ColumnBase:
        """Identify NaN values in a Column."""
        return as_column(False, length=len(self))

    def notnan(self) -> ColumnBase:
        """Identify non-NaN values in a Column."""
        return as_column(True, length=len(self))

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column."""
        if not self.has_nulls(include_nan=False):
            return as_column(False, length=len(self))

        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.unary.is_null(self.plc_column), np.dtype(np.bool_)
            )

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column."""
        if not self.has_nulls(include_nan=False):
            result = as_column(True, length=len(self))
        else:
            with self.access(mode="read", scope="internal"):
                result = ColumnBase.create(
                    plc.unary.is_valid(self.plc_column), np.dtype(np.bool_)
                )

        if cudf.get_option("mode.pandas_compatible"):
            return result

        return ColumnBase.create(
            result.plc_column,
            get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
        )

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
            return cast("Self", column_empty(row_count=0, dtype=self.dtype))

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if indices.dtype.kind not in {"u", "i"}:
            indices = indices.astype(SIZE_TYPE_DTYPE)
        GatherMap(indices, len(self), nullify=not check_bounds or nullify)
        gathered = copying.gather([self], indices, nullify=nullify)[0]  # type: ignore[arg-type]
        return cast(
            "Self",
            ColumnBase.create(gathered, self.dtype),
        )

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

    def as_mask(self) -> tuple[Buffer, int]:
        """Convert booleans to bitmask

        Returns
        -------
        tuple[Buffer, int]
            The mask buffer and the null count (number of False values).
        """
        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        with self.access(mode="read", scope="internal"):
            mask, null_count = plc.transform.bools_to_mask(self.plc_column)
            return as_buffer(mask), null_count

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

    def contains(self, other: ColumnBase) -> ColumnBase:
        """
        Check whether column contains multiple values.

        Parameters
        ----------
        other : Column
            A column of values to search for
        """
        with access_columns(self, other, mode="read", scope="internal") as (
            self,
            other,
        ):
            return ColumnBase.create(
                plc.search.contains(
                    self.plc_column,
                    other.plc_column,
                ),
                np.dtype(np.bool_),
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
        with self.access(mode="read", scope="internal"):
            plc_table = plc.sorting.sort(
                plc.Table([self.plc_column]),
                order[0],
                order[1],
            )
            return cast(
                "Self",
                ColumnBase.create(plc_table.columns()[0], self.dtype),
            )

    def distinct_count(self, dropna: bool = True) -> int:
        """Get the (null-aware) number of distinct values in this column."""
        try:
            return self._distinct_count[dropna]
        except KeyError:
            with self.access(mode="read", scope="internal"):
                result = plc.stream_compaction.distinct_count(
                    self.plc_column,
                    plc.types.NullPolicy.EXCLUDE
                    if dropna
                    else plc.types.NullPolicy.INCLUDE,
                    plc.types.NanPolicy.NAN_IS_NULL
                    if dropna
                    else plc.types.NanPolicy.NAN_IS_VALID,
                )
            self._distinct_count[dropna] = result
            return result

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        raise NotImplementedError()

    def cast(self, dtype: DtypeObj) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            result = type(self).from_pylibcudf(
                plc.unary.cast(self.plc_column, dtype_to_pylibcudf_type(dtype))
            )
            # Adjust decimal result: in pandas compat mode with non-decimal target,
            # preserve the target dtype wrapper; otherwise update precision from target
            if isinstance(result.dtype, DecimalDtype):
                if cudf.get_option(
                    "mode.pandas_compatible"
                ) and not isinstance(dtype, DecimalDtype):
                    result._dtype = dtype
                else:
                    result.dtype.precision = dtype.precision  # type: ignore[union-attr]
            if (
                cudf.get_option("mode.pandas_compatible")
                and result.dtype != dtype
            ):
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
        return codes.set_mask(self.mask, self.null_count)._with_type_metadata(
            dtype
        )  # type: ignore[return-value]

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        raise NotImplementedError()

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        raise NotImplementedError()

    def as_interval_column(self, dtype: IntervalDtype) -> IntervalColumn:
        raise NotImplementedError()

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        raise NotImplementedError()

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        raise NotImplementedError()

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalBaseColumn:
        raise NotImplementedError()

    def apply_boolean_mask(self, mask: ColumnBase) -> ColumnBase:
        if mask.dtype.kind != "b":
            raise ValueError("boolean_mask is not boolean type.")

        with access_columns(self, mask, mode="read", scope="internal") as (
            col,
            mask_col,
        ):
            plc_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table([col.plc_column]),
                mask_col.plc_column,
            )
            return ColumnBase.create(plc_table.columns()[0], self.dtype)

    def argsort(
        self,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> NumericalColumn:
        if (ascending and self.is_monotonic_increasing) or (
            not ascending and self.is_monotonic_decreasing
        ):
            return cast(
                "cudf.core.column.NumericalColumn", as_column(range(len(self)))
            )
        elif (ascending and self.is_monotonic_decreasing) or (
            not ascending and self.is_monotonic_increasing
        ):
            return cast(
                "cudf.core.column.NumericalColumn",
                as_column(range(len(self) - 1, -1, -1)),
            )
        else:
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.from_pylibcudf(
                    sorting.order_by(
                        [self], [ascending], [na_position], stable=True
                    )
                ),
            )

    def __arrow_array__(self, type: pa.DataType | None = None) -> None:
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        # pandas produces non-writeable numpy arrays when CoW is enabled, which cupy
        # does not support. Our options are either to copy the data preemptively to
        # avoid handing out a pointer that allows modification of the data, or to allow
        # such modification even though it doesn't match pandas default behavior. Since
        # numpy arrays can be trivially made writeable by just changing their flag,
        # allowing modification here seems like the better option since otherwise we
        # would forbid modification altogether. Moreover, preemptive copying would
        # increase memory pressure unnecessarily.
        data_buf = self.data
        if data_buf is None:
            raise ValueError(
                "__cuda_array_interface__ not supported for columns with no data buffer"
            )
        self._exposed_buffers.add(data_buf)
        with data_buf.access(mode="read", scope="external"):
            output = {
                "shape": (len(self),),
                "strides": (self.dtype.itemsize,),
                "typestr": self.dtype.str,
                "data": (
                    data_buf.ptr + self.offset * self.dtype.itemsize,
                    False,
                ),
                "version": 3,
            }
        if self.nullable and self.has_nulls():
            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            mask = self.mask
            assert mask is not None
            with mask.access(mode="read", scope="external"):
                output["mask"] = mask
                self._exposed_buffers.add(mask)
        return output

    def __array_ufunc__(
        self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any
    ) -> ColumnBase:
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    def __invert__(self) -> ColumnBase:
        raise TypeError(
            f"Operation `~` not supported on {self.dtype.type.__name__}"
        )

    def searchsorted(
        self,
        value: ColumnBase,
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
        return cast(
            "Self",
            ColumnBase.from_pylibcudf(
                sorting.search_sorted(
                    [self],
                    [value],
                    side=side,
                    ascending=[ascending],
                    na_position=[na_position],
                )
            ),
        )

    def unique(self) -> Self:
        """
        Get unique values in the data
        """
        if self.is_unique:
            return self.copy()
        else:
            with self.access(mode="read", scope="internal"):
                plc_table = plc.stream_compaction.stable_distinct(
                    plc.Table([self.plc_column]),
                    [0],
                    plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                    plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL,
                )
                return cast(
                    "Self",
                    ColumnBase.create(plc_table.columns()[0], self.dtype),
                )

    def _is_sliced(self) -> bool:
        """
        Check if this column is a sliced view of a larger buffer.

        A column is considered sliced if:
        - It has a non-zero offset, OR
        - Its logical size is less than the base buffer size

        Returns
        -------
        bool
            True if the column is sliced, False otherwise
        """
        if self.offset != 0:
            return True

        # Check if size is less than base buffer size
        # Note: Some dtypes (like CategoricalDtype) don't have itemsize
        if self.data is not None and hasattr(self.dtype, "itemsize"):
            base_size_elements = self.data.size // self.dtype.itemsize
            if self.size < base_size_elements:
                return True

        return False

    def _compact_for_transfer(self) -> ColumnBase:
        """
        Create a compact copy of this column for network transfer.

        If the column is sliced (offset != 0 or size < base buffer size),
        this creates a copy that contains only the data that's actually
        needed, not the entire base buffer. This significantly reduces
        network transfer overhead when serializing sliced columns.

        Returns
        -------
        ColumnBase
            A compacted column (or self if not sliced)
        """
        if self._is_sliced():
            # Check if data is spilled - if so, don't compact it
            # Spilled data is already on CPU and calling copy() would
            # unnecessarily bring it back to GPU and lose the spill state
            if self.data is not None and hasattr(self.data, "owner"):
                if (
                    hasattr(self.data.owner, "is_spilled")
                    and self.data.owner.is_spilled
                ):
                    return self

            # Create a copy to compact the data - only the actual slice
            # will be copied, not the entire base buffer
            return self.copy()
        return self

    @staticmethod
    def _serialize_plc_column_recursive(
        plc_col: plc.Column,
    ) -> tuple[dict, list]:
        """Recursively serialize a plc.Column and all its children."""
        header: dict[Any, Any] = {}
        frames = []

        if (plc_data := plc_col.data()) is not None:
            header["data"], data_frames = cast(
                "Buffer", plc_data
            ).device_serialize()
            frames.extend(data_frames)

        if (plc_mask := plc_col.null_mask()) is not None:
            header["mask"], mask_frames = cast(
                "Buffer", plc_mask
            ).device_serialize()
            frames.extend(mask_frames)

        header["plc_type"] = pickle.dumps(plc_col.type())
        header["size"] = plc_col.size()
        header["offset"] = plc_col.offset()
        header["null_count"] = plc_col.null_count()

        # Recursively serialize children
        if plc_children := plc_col.children():
            child_subheaders = []
            for plc_child in plc_children:
                child_header, child_frames = (
                    ColumnBase._serialize_plc_column_recursive(plc_child)
                )
                child_subheaders.append(child_header)
                frames.extend(child_frames)
            header["subheaders"] = child_subheaders

        header["frame_count"] = len(frames)
        return header, frames

    @staticmethod
    def _unpack_buffer(header: dict, frames: list) -> tuple[Any, list]:
        """Unpack a serialized buffer from frames, returning (object, remaining_frames)."""
        count = header["frame_count"]
        return Serializable.device_deserialize(header, frames[:count]), frames[
            count:
        ]

    @classmethod
    def _deserialize_plc_column_recursive(
        cls, header: dict, frames: list
    ) -> tuple[plc.Column, list]:
        """Recursively deserialize a plc.Column and all its children."""
        # Deserialize data and mask buffers
        data = None
        if "data" in header:
            data, frames = cls._unpack_buffer(header["data"], frames)

        mask = None
        if "mask" in header:
            mask, frames = cls._unpack_buffer(header["mask"], frames)

        children = []
        for child_header in header.get("subheaders", []):
            child_plc_col, frames = cls._deserialize_plc_column_recursive(
                child_header, frames
            )
            children.append(child_plc_col)

        try:
            plc_type = pickle.loads(header["plc_type"])
        except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as e:
            raise ValueError(f"Failed to deserialize plc_type: {e}") from e

        return plc.Column(
            plc_type,
            header["size"],
            data,
            mask,
            header["null_count"],
            header.get("offset", 0),
            children,
            validate=False,
        ), frames

    def serialize(self) -> tuple[dict, list]:
        """Serialize column to header dict and frames list.

        Produces nested metadata header and flattened list of buffers (frames).
        Each header has frame_count indicating frames consumed during deserialization.
        """
        # Compact sliced columns before serialization to avoid transferring
        # entire base buffers over the network. This is critical for
        # performance when working with distributed systems like Dask.
        # Only affects sliced columns (offset != 0 or size < base buffer size).
        col_to_serialize = self._compact_for_transfer()

        header: dict[Any, Any] = {}
        frames = []

        # Serialize dtype
        try:
            header["dtype"], dtype_frames = (
                col_to_serialize.dtype.device_serialize()  # type: ignore[union-attr]
            )
            frames.extend(dtype_frames)
            header["dtype-is-cudf-serialized"] = True
        except AttributeError:
            header["dtype"] = (
                pickle.dumps(col_to_serialize.dtype)
                if is_pandas_nullable_extension_dtype(col_to_serialize.dtype)
                else col_to_serialize.dtype.str
            )
            header["dtype-is-cudf-serialized"] = False

        # Serialize entire plc_column (data, mask, children)
        plc_header, plc_frames = self._serialize_plc_column_recursive(
            col_to_serialize.plc_column
        )
        frames.extend(plc_frames)
        header["plc_column"] = plc_header

        header["size"] = col_to_serialize.size
        header["frame_count"] = len(frames)
        header["offset"] = col_to_serialize.offset
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> ColumnBase:
        assert header["frame_count"] == len(frames), (
            f"Expected {header['frame_count']} frames, got {len(frames)}"
        )

        # Deserialize dtype
        if header["dtype-is-cudf-serialized"]:
            dtype, frames = cls._unpack_buffer(header["dtype"], frames)
        else:
            try:
                dtype = np.dtype(header["dtype"])
            except TypeError:
                dtype = pickle.loads(header["dtype"])

        # Deserialize plc_column recursively
        plc_column, frames = cls._deserialize_plc_column_recursive(
            header["plc_column"], frames
        )
        assert len(frames) == 0, (
            f"{len(frames)} frame(s) remaining after deserialization"
        )
        return ColumnBase.create(plc_column, dtype)

    def unary_operator(self, unaryop: str) -> ColumnBase:
        raise TypeError(
            f"Operation {unaryop} not supported for dtype {self.dtype}."
        )

    def nans_to_nulls(self: Self) -> Self:
        """Convert NaN to NA."""
        return self

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls(include_nan=False)

    def _reduction_result_dtype(self, op: str) -> DtypeObj:
        """
        Determine the correct dtype to pass to libcudf based on
        the input dtype, data dtype, and specific reduction op
        """
        if op in {"any", "all"}:
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

        def _return_sentinel_column() -> NumericalColumn:
            # TODO: Validate that dtype is numeric type
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                as_column(na_sentinel, dtype=dtype, length=len(self)),
            )

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
            plc.Table([self.plc_column]),
            plc.Table([cats.plc_column]),
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
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            ColumnBase.from_pylibcudf(plc_codes).fillna(na_sentinel),
        )

    def copy_if_else(
        self, other: Self | plc.Scalar, boolean_mask: NumericalColumn
    ) -> Self:
        with access_columns(
            self, other, boolean_mask, mode="read", scope="internal"
        ) as (self, other, boolean_mask):
            return cast(
                "Self",
                ColumnBase.create(
                    plc.copying.copy_if_else(
                        self.plc_column,
                        other
                        if isinstance(other, plc.Scalar)
                        else other.plc_column,
                        boolean_mask.plc_column,
                    ),
                    self.dtype,
                ),
            )

    def split_by_offsets(
        self, offsets: list[int]
    ) -> Generator[Self, None, None]:
        for cols in copying.columns_split([self], offsets):
            for col in cols:
                yield cast(
                    "Self",
                    ColumnBase.create(col, self.dtype),
                )

    def one_hot_encode(self, categories: ColumnBase) -> Generator[ColumnBase]:
        with access_columns(
            self, categories, mode="read", scope="internal"
        ) as (self, categories):
            plc_table = plc.transform.one_hot_encode(
                self.plc_column,
                categories.plc_column,
            )
            return (
                type(self).from_pylibcudf(col) for col in plc_table.columns()
            )

    def _scan(self, op: str, inclusive: bool = True, **kwargs: Any) -> Self:
        """Private method for scan operations. Called by mixin-generated methods."""
        # `inclusive` controls scan type, not passed to aggregation
        with self.access(mode="read", scope="internal"):
            plc_result = plc.reduce.scan(
                self.plc_column,
                aggregation.make_aggregation(op, kwargs).plc_obj,
                plc.reduce.ScanType.INCLUSIVE
                if inclusive
                else plc.reduce.ScanType.EXCLUSIVE,
            )
        return cast("Self", ColumnBase.create(plc_result, self.dtype))

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Private method for reduction operations. Called by mixin-generated methods."""
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        # Early return if we can return NaN
        if self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        # Handle skipna by converting nans to nulls and potentially dropping
        col = self.nans_to_nulls() if skipna else self
        if col.has_nulls():
            if skipna:
                col = col.dropna()
            else:
                return _get_nan_for_dtype(self.dtype)

        # Handle min_count
        if min_count > 0:
            valid_count = len(col) - col.null_count
            if valid_count < min_count:
                return _get_nan_for_dtype(self.dtype)

        # Compute reduction result dtype
        col_dtype = col._reduction_result_dtype(op)

        # Handle empty case
        if len(col) <= col.null_count:
            if op == "sum" or op == "sum_of_squares":
                return col_dtype.type(0)
            if op == "product":
                return col_dtype.type(1)
            return _get_nan_for_dtype(col_dtype)

        # Perform the actual reduction
        with col.access(mode="read", scope="internal"):
            plc_scalar = plc.reduce.reduce(
                col.plc_column,
                aggregation.make_aggregation(op, kwargs).plc_obj,
                dtype_to_pylibcudf_type(col_dtype),
            )
            result_col = ColumnBase.create(
                plc.Column.from_scalar(plc_scalar, 1), col_dtype
            )
            # Hook for subclasses (e.g., DecimalBaseColumn adjusts precision)
            result_col = col._adjust_reduce_result(
                result_col, op, col_dtype, plc_scalar
            )
        return result_col.element_indexing(0)

    def _adjust_reduce_result(
        self,
        result_col: ColumnBase,
        op: str,
        col_dtype: DtypeObj,
        plc_scalar: plc.Scalar,
    ) -> ColumnBase:
        """Hook for subclasses to adjust reduction result."""
        return result_col

    def minmax(self) -> tuple[ScalarLike, ScalarLike]:
        with self.access(mode="read", scope="internal"):
            min_val, max_val = plc.reduce.minmax(self.plc_column)
            return (
                type(self)
                .from_pylibcudf(plc.Column.from_scalar(min_val, 1))
                .element_indexing(0),
                type(self)
                .from_pylibcudf(plc.Column.from_scalar(max_val, 1))
                .element_indexing(0),
            )

    def rank(
        self,
        *,
        method: plc.aggregation.RankMethod,
        column_order: plc.types.Order,
        null_handling: plc.types.NullPolicy,
        null_precedence: plc.types.NullOrder,
        pct: bool,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                "Self",
                type(self).from_pylibcudf(
                    plc.sorting.rank(
                        self.plc_column,
                        method,
                        column_order,
                        null_handling,
                        null_precedence,
                        pct,
                    )
                ),
            )

    def label_bins(
        self,
        *,
        left_edge: Self,
        left_inclusive: bool,
        right_edge: Self,
        right_inclusive: bool,
    ) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                type(self).from_pylibcudf(
                    plc.labeling.label_bins(
                        self.plc_column,
                        left_edge.plc_column,
                        plc.labeling.Inclusive.YES
                        if left_inclusive
                        else plc.labeling.Inclusive.NO,
                        right_edge.plc_column,
                        plc.labeling.Inclusive.YES
                        if right_inclusive
                        else plc.labeling.Inclusive.NO,
                    )
                ),
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
                    # can_cast_safely should imply this is safe
                    np.min_scalar_type(other)  # type: ignore[arg-type]
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
        result = casted_col.copy_if_else(casted_other, cond)  # type: ignore[arg-type]
        return ColumnBase.create(result.plc_column, self.dtype)


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
    if (
        is_struct := isinstance(dtype, (StructDtype, IntervalDtype))
    ) or isinstance(dtype, ListDtype):
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
            else plc.null_mask.create_null_mask(
                row_count, plc.types.MaskState.ALL_NULL
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
                [child.plc_column for child in children],
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


def check_invalid_array(shape: tuple, dtype: np.dtype) -> None:
    """Invalid ndarrays properties that are not supported"""
    if len(shape) > 1:
        raise ValueError("Data must be 1-dimensional")
    elif dtype == "float16":
        raise TypeError("Unsupported type float16")


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
            if isinstance(arbitrary.dtype, pd.IntervalDtype):
                # Wrap StructColumn as IntervalColumn with proper metadata
                result = result._with_type_metadata(
                    IntervalDtype(
                        subtype=arbitrary.dtype.subtype,
                        closed=arbitrary.dtype.closed,
                    )
                )
            elif (
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
                column = ColumnBase.from_cuda_array_interface(arbitrary)
                if nan_as_null is not False:
                    column = column.nans_to_nulls()
                if dtype is not None:
                    column = column.astype(dtype)
                return column
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
                raise MixedTypeError(
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
    else:
        # Try to convert to memoryview
        try:
            view = memoryview(arbitrary)
            return as_column(
                np.asarray(view), dtype=dtype, nan_as_null=nan_as_null
            )
        except TypeError:
            pass

        # Memoryview failed, check if it's iterable
        if not isinstance(arbitrary, (Iterable, Sequence)):
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
        if isinstance(dtype, cudf.Decimal128Dtype):
            pa_type = pa.decimal128(
                precision=dtype.precision, scale=dtype.scale
            )
            column_class = cudf.core.column.Decimal128Column
        elif isinstance(dtype, cudf.Decimal64Dtype):
            pa_type = pa.decimal64(
                precision=dtype.precision, scale=dtype.scale
            )
            column_class = cudf.core.column.Decimal64Column
        elif isinstance(dtype, cudf.Decimal32Dtype):
            pa_type = pa.decimal32(
                precision=dtype.precision, scale=dtype.scale
            )
            column_class = cudf.core.column.Decimal32Column
        else:
            raise NotImplementedError(f"{dtype} not implemented")
        data = pa.array(arbitrary, type=pa_type)
        return column_class.from_arrow(data)
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
                ser = pd.Series(
                    arbitrary, dtype=pd.CategoricalDtype(ordered=dtype.ordered)
                )
                if dtype.categories is not None:
                    ser = ser.cat.set_categories(
                        dtype.categories, ordered=dtype.ordered
                    )
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
                isinstance(element, (pd.Timestamp, pd.Timedelta, pd.Interval))
                or element is pd.NaT
            ):
                # TODO: Remove this after
                # https://github.com/apache/arrow/issues/26492
                # is fixed.
                # Note: pd.Interval also requires pandas Series conversion
                # because PyArrow cannot infer interval type from raw list
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
                "MutableSequence[cudf.core.column.categorical.CategoricalColumn]",
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
    with access_columns(  # type: ignore[assignment]
        *objs_with_len, mode="read", scope="internal"
    ) as objs_with_len:
        return ColumnBase.create(
            plc.concatenate.concatenate(
                [col.plc_column for col in objs_with_len]
            ),
            objs_with_len[0].dtype,
        )
