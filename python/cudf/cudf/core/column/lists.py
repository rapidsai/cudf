# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.dtype.validators import is_dtype_obj_list
from cudf.core.dtypes import ListDtype
from cudf.core.missing import NA
from cudf.utils.dtypes import get_dtype_of_same_kind
from cudf.utils.scalar import (
    maybe_nested_pa_scalar_to_py,
    pa_scalar_to_plc_scalar,
)
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cudf._typing import ColumnBinaryOperand, ColumnLike, DtypeObj
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


class ListColumn(ColumnBase):
    _VALID_BINARY_OPERATIONS = {"__add__", "__radd__", "__eq__", "__ne__"}
    _VALID_PLC_TYPES = {plc.TypeId.LIST}

    @classmethod
    def _validate_args(  # type: ignore[override]
        cls, plc_column: plc.Column, dtype: ListDtype
    ) -> tuple[plc.Column, ListDtype]:
        plc_column, dtype = super()._validate_args(plc_column, dtype)  # type: ignore[assignment]
        if not is_dtype_obj_list(dtype):
            raise ValueError("dtype must be a cudf.ListDtype")

        child = plc_column.list_view().child()
        try:
            ColumnBase._validate_dtype_recursively(child, dtype.element_type)
        except ValueError as e:
            raise ValueError(
                f"List element type validation failed: {e}"
            ) from e

        return plc_column, dtype

    def _get_sliced_child(self) -> ColumnBase:
        """Get a child column properly sliced to match the parent's view."""
        sliced_plc_col = self.plc_column.list_view().get_sliced_child()
        assert isinstance(self.dtype, ListDtype)
        return ColumnBase.create(sliced_plc_col, self.dtype.element_type)

    def _prep_pandas_compat_repr(self) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        # TODO: handle if self.has_nulls(): case
        return self

    def element_indexing(self, index: int) -> list:
        result = super().element_indexing(index)
        if isinstance(result, pa.Scalar):
            py_element = maybe_nested_pa_scalar_to_py(result)
            return self.dtype._recursively_replace_fields(py_element)  # type: ignore[union-attr]
        return result

    def _cast_setitem_value(self, value: Any) -> plc.Scalar:
        if isinstance(value, list) or value is None:
            return pa_scalar_to_plc_scalar(
                pa.scalar(value, type=self.dtype.to_arrow())  # type: ignore[union-attr]
            )
        elif value is NA or value is None:
            return pa_scalar_to_plc_scalar(
                pa.scalar(None, type=self.dtype.to_arrow())  # type: ignore[union-attr]
            )
        else:
            raise ValueError(f"Can not set {value} into ListColumn")

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        # Lists only support __add__, which concatenates lists.
        reflect, op = self._check_reflected_op(op)
        if not isinstance(other, type(self)):
            return NotImplemented
        if isinstance(other.dtype, ListDtype):
            if op == "__add__":
                return self.concatenate_rows([other])
            else:
                raise NotImplementedError(
                    "Lists concatenation for this operation is not yet"
                    "supported"
                )
        else:
            raise TypeError("can only concatenate list to list")

    @property
    def elements(self) -> ColumnBase:
        """Column containing the elements of each list (may itself be a ListColumn)"""
        return self._get_sliced_child()

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Lists are not yet supported via `__cuda_array_interface__`"
        )

    def _with_type_metadata(self: Self, dtype: DtypeObj) -> Self:
        if isinstance(dtype, ListDtype):
            self._dtype = dtype
        elif isinstance(dtype, pd.ArrowDtype) and isinstance(
            dtype.pyarrow_dtype, pa.ListType
        ):
            self._dtype = dtype
        return self

    def copy(self, deep: bool = True) -> Self:
        # Since list columns are immutable, both deep and shallow copies share
        # the underlying device data and mask.
        return super().copy(deep=False)

    def leaves(self) -> ColumnBase:
        if isinstance(self.elements, ListColumn):
            return self.elements.leaves()
        else:
            return self.elements

    @classmethod
    def from_sequences(cls, arbitrary: Sequence[ColumnLike]) -> Self:
        """
        Create a list column for list of column-like sequences
        """
        data_col = column_empty(0)
        mask_bools = []
        offset_vals = [0]
        offset = 0

        # Build Data, Mask & Offsets
        for data in arbitrary:
            if is_na_like(data):
                mask_bools.append(False)
                offset_vals.append(offset)
            else:
                mask_bools.append(True)
                data_col = data_col.append(as_column(data))
                offset += len(data)
                offset_vals.append(offset)

        offset_col = plc.Column.from_iterable_of_py(
            offset_vals, dtype=plc.types.SIZE_TYPE
        )
        data_plc_col = data_col.plc_column
        mask, null_count = plc.transform.bools_to_mask(
            plc.Column.from_iterable_of_py(mask_bools)
        )
        plc_column = plc.Column(
            plc.DataType(plc.TypeId.LIST),
            len(offset_vals) - 1,
            None,
            mask,
            null_count,
            0,
            [offset_col, data_plc_col],
        )
        return cast(
            "Self",
            ColumnBase.create(plc_column, ListDtype(data_col.dtype)),
        )

    @cached_property
    def _string_separators(self) -> plc.Column:
        # Separator strings to match the Python format
        return plc.Column.from_iterable_of_py(
            [", ", "[", "]"], dtype=plc.DataType(plc.TypeId.STRING)
        )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        """
        Create a strings column from a list column
        """
        if cudf.get_option("mode.pandas_compatible"):
            if isinstance(dtype, np.dtype) and dtype.kind == "O":
                raise TypeError(
                    f"Cannot cast a list from {self.dtype} to {dtype}"
                )
        lc = self._transform_leaves(
            lambda col, dtype: col.as_string_column(dtype), dtype
        )

        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_lists.format_list_column(
                lc.plc_column,
                pa_scalar_to_plc_scalar(pa.scalar("None")),
                self._string_separators,
            )
            return cast(
                "cudf.core.column.string.StringColumn",
                ColumnBase.create(plc_column, dtype),
            )

    def _transform_leaves(
        self, func: Callable[[ColumnBase, DtypeObj], ColumnBase], *args: Any
    ) -> Self:
        """
        Return a new column like Self but with func applied to the last leaf column.
        """
        # Store metadata for reconstruction: (plc_column, list_view)
        # We need to keep the full plc_column for accessing size, mask, null_count, offset
        leaf_queue: list[plc.Column] = []
        curr_plc_col: plc.Column = self.plc_column
        leaf_dtype = self.dtype

        while curr_plc_col.type().id() == plc.TypeId.LIST:
            leaf_queue.append(curr_plc_col)
            curr_plc_col = curr_plc_col.list_view().child()
            leaf_dtype = cast(ListDtype, leaf_dtype).element_type

        # Apply the transformation to the leaf column
        # TODO: For now we convert plc.Column to ColumnBase for the func, then back to
        # plc.Column, but we should be able to eventually avoid this double conversion.
        leaf_col_base = ColumnBase.create(curr_plc_col, leaf_dtype)
        transformed_leaf = func(leaf_col_base, *args)
        plc_leaf_col = transformed_leaf.plc_column

        # Rebuild the list column hierarchy from leaf back to root
        result_dtype = transformed_leaf.dtype
        while leaf_queue:
            parent_plc_col = leaf_queue.pop()
            offsets = parent_plc_col.list_view().offsets()
            # parent_plc_col.null_mask() is a Span which is Span-compliant
            plc_leaf_col = plc.Column(
                plc.DataType(plc.TypeId.LIST),
                parent_plc_col.size(),
                None,
                parent_plc_col.null_mask(),
                parent_plc_col.null_count(),
                parent_plc_col.offset(),
                [offsets, plc_leaf_col],
            )
            result_dtype = ListDtype(result_dtype)
        return cast(
            "Self",
            ColumnBase.create(plc_leaf_col, result_dtype),
        )

    @property
    def element_type(self) -> DtypeObj:
        """
        Returns the element type of the list column.
        """
        if isinstance(self.dtype, ListDtype):
            return self.dtype.element_type
        else:
            return get_dtype_of_same_kind(
                self.dtype,
                self.dtype.pyarrow_dtype.value_type.to_pandas_dtype(),  # type: ignore[union-attr]
            )

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        else:
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    def count_elements(self) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.count_elements(self.plc_column),
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
            )

    def distinct(self, nulls_equal: bool, nans_all_equal: bool) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.distinct(
                    self.plc_column,
                    (
                        plc.types.NullEquality.EQUAL
                        if nulls_equal
                        else plc.types.NullEquality.UNEQUAL
                    ),
                    (
                        plc.types.NanEquality.ALL_EQUAL
                        if nans_all_equal
                        else plc.types.NanEquality.UNEQUAL
                    ),
                ),
                self.dtype,
            )

    def sort_lists(
        self, ascending: bool, na_position: Literal["first", "last"]
    ) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.sort_lists(
                    self.plc_column,
                    plc.types.Order.ASCENDING
                    if ascending
                    else plc.types.Order.DESCENDING,
                    (
                        plc.types.NullOrder.BEFORE
                        if na_position == "first"
                        else plc.types.NullOrder.AFTER
                    ),
                    False,
                ),
                self.dtype,
            )

    def extract_element_scalar(self, index: int) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.extract_list_element(
                    self.plc_column,
                    index,
                ),
                self.element_type,
            )

    def extract_element_column(self, index: ColumnBase) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.extract_list_element(
                    self.plc_column,
                    index.plc_column,
                ),
                self.element_type,
            )

    def contains_scalar(self, search_key: pa.Scalar) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.contains(
                    self.plc_column,
                    pa_scalar_to_plc_scalar(search_key),
                ),
                get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
            )

    def index_of_scalar(self, search_key: pa.Scalar) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.index_of(
                    self.plc_column,
                    pa_scalar_to_plc_scalar(search_key),
                    plc.lists.DuplicateFindOption.FIND_FIRST,
                ),
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
            )

    def index_of_column(self, search_keys: ColumnBase) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.index_of(
                    self.plc_column,
                    search_keys.plc_column,
                    plc.lists.DuplicateFindOption.FIND_FIRST,
                ),
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
            )

    def concatenate_rows(self, other_columns: list[ColumnBase]) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.concatenate_rows(
                    plc.Table(
                        [
                            col.plc_column
                            for col in itertools.chain([self], other_columns)
                        ]
                    )
                ),
                self.dtype,
            )

    def concatenate_list_elements(self, dropna: bool) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.concatenate_list_elements(
                    self.plc_column,
                    plc.lists.ConcatenateNullPolicy.IGNORE
                    if dropna
                    else plc.lists.ConcatenateNullPolicy.NULLIFY_OUTPUT_ROW,
                ),
                self.element_type,
            )

    def segmented_gather(self, gather_map: ColumnBase) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.segmented_gather(
                    self.plc_column,
                    gather_map.plc_column,
                ),
                self.dtype,
            )

    def join_list_elements(
        self,
        separator: str | StringColumn,
        sep_na_rep: str,
        string_na_rep: str,
        result_dtype: DtypeObj,
    ) -> StringColumn:
        with self.access(mode="read", scope="internal"):
            if isinstance(separator, str):
                sep: plc.Scalar | plc.Column = pa_scalar_to_plc_scalar(
                    pa.scalar(separator)
                )
            else:
                sep = separator.plc_column
            plc_column = plc.strings.combine.join_list_elements(
                self.plc_column,
                sep,
                pa_scalar_to_plc_scalar(pa.scalar(sep_na_rep)),
                pa_scalar_to_plc_scalar(pa.scalar(string_na_rep)),
                plc.strings.combine.SeparatorOnNulls.YES,
                plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
            )
            return cast(
                "cudf.core.column.string.StringColumn",
                ColumnBase.create(plc_column, result_dtype),
            )

    def minhash_ngrams(
        self,
        width: int,
        seed: int | np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint32 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                    )
                seed = np.uint32(seed)
            return cast(
                "Self",
                ColumnBase.create(
                    plc.nvtext.minhash.minhash_ngrams(
                        self.plc_column,
                        width,
                        seed,
                        a.plc_column,
                        b.plc_column,
                    ),
                    ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32))
                    ),
                ),
            )

    def minhash64_ngrams(
        self,
        width: int,
        seed: int | np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint64 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint64).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint64).max}]"
                    )
                seed = np.uint64(seed)
            return cast(
                "Self",
                ColumnBase.create(
                    plc.nvtext.minhash.minhash64_ngrams(
                        self.plc_column,
                        width,
                        seed,
                        a.plc_column,
                        b.plc_column,
                    ),
                    ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint64))
                    ),
                ),
            )
