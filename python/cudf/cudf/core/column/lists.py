# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.column.numerical import NumericalColumn
from cudf.core.dtypes import ListDtype
from cudf.core.missing import NA
from cudf.utils.dtypes import (
    get_dtype_of_same_kind,
    is_dtype_obj_list,
)
from cudf.utils.scalar import (
    maybe_nested_pa_scalar_to_py,
    pa_scalar_to_plc_scalar,
)
from cudf.utils.utils import _is_null_host_scalar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from typing_extensions import Self

    from cudf._typing import ColumnBinaryOperand, ColumnLike, DtypeObj
    from cudf.core.column.string import StringColumn


class ListColumn(ColumnBase):
    _VALID_BINARY_OPERATIONS = {"__add__", "__radd__"}
    _VALID_PLC_TYPES = {plc.TypeId.LIST}

    def __init__(
        self,
        plc_column: plc.Column,
        size: int,
        dtype: ListDtype,
        offset: int,
        null_count: int,
        exposed: bool,
    ) -> None:
        if (
            not cudf.get_option("mode.pandas_compatible")
            and not isinstance(dtype, ListDtype)
        ) or (
            cudf.get_option("mode.pandas_compatible")
            and not is_dtype_obj_list(dtype)
        ):
            raise ValueError("dtype must be a cudf.ListDtype")
        super().__init__(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )

    def _get_children_from_pylibcudf_column(
        self,
        plc_column: plc.Column,
        dtype: ListDtype,  # type: ignore[override]
        exposed: bool,
    ) -> tuple[ColumnBase, ColumnBase]:
        children = super()._get_children_from_pylibcudf_column(
            plc_column, dtype, exposed
        )
        return (
            children[0],
            children[1]._with_type_metadata(dtype.element_type),
        )

    def _prep_pandas_compat_repr(self) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        # TODO: handle if self.has_nulls(): case
        return self

    @cached_property
    def memory_usage(self) -> int:
        n = super().memory_usage
        child0_size = (self.size + 1) * self.base_children[0].dtype.itemsize
        current_base_child = self.base_children[1]
        current_offset = self.offset
        n += child0_size
        while type(current_base_child) is ListColumn:
            child0_size = (
                current_base_child.size + 1 - current_offset
            ) * current_base_child.base_children[0].dtype.itemsize
            n += child0_size
            current_offset_col = current_base_child.base_children[0]
            if not len(current_offset_col):
                # See https://github.com/rapidsai/cudf/issues/16164 why
                # offset column can be uninitialized
                break
            current_offset = current_offset_col.element_indexing(
                current_offset
            )
            current_base_child = current_base_child.base_children[1]

        n += (
            current_base_child.size - current_offset
        ) * current_base_child.dtype.itemsize

        if current_base_child.nullable:
            n += plc.null_mask.bitmask_allocation_size_bytes(
                current_base_child.size
            )
        return n

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

    @property
    def base_size(self) -> int:
        # in some cases, libcudf will return an empty ListColumn with no
        # indices; in these cases, we must manually set the base_size to 0 to
        # avoid it being negative
        return max(0, len(self.base_children[0]) - 1)

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
        """
        Column containing the elements of each list (may itself be a
        ListColumn)
        """
        return self.children[1]

    @property
    def offsets(self) -> NumericalColumn:
        """
        Integer offsets to elements specifying each row of the ListColumn
        """
        return cast(NumericalColumn, self.children[0])

    def to_arrow(self) -> pa.Array:
        offsets = self.offsets.to_arrow()
        elements = (
            pa.nulls(len(self.elements))
            if len(self.elements) == self.elements.null_count
            else self.elements.to_arrow()
        )
        pa_type = pa.list_(elements.type)

        if self.nullable:
            nbuf = pa.py_buffer(self.mask.memoryview())  # type: ignore[union-attr]
            buffers = [nbuf, offsets.buffers()[1]]
        else:
            buffers = list(offsets.buffers())
        return pa.ListArray.from_buffers(
            pa_type,
            len(self),
            # PyArrow stubs are too strict - from_buffers should accept None for missing buffers
            buffers,  # type: ignore[arg-type]
            children=[elements],
        )

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Lists are not yet supported via `__cuda_array_interface__`"
        )

    def _with_type_metadata(self: Self, dtype: DtypeObj) -> Self:
        if isinstance(dtype, ListDtype):
            elements = self.base_children[1]._with_type_metadata(
                dtype.element_type
            )
            new_children = [
                self.plc_column.children()[0],
                elements.to_pylibcudf(mode="read"),
            ]
            new_plc_column = plc.Column(
                plc.DataType(plc.TypeId.LIST),
                self.plc_column.size(),
                self.plc_column.data(),
                self.plc_column.null_mask(),
                self.plc_column.null_count(),
                self.plc_column.offset(),
                new_children,
            )
            return type(self)(
                plc_column=new_plc_column,
                size=self.size,
                dtype=dtype,
                offset=self.offset,
                null_count=self.null_count,
                exposed=False,
            )
        # For pandas dtypes, store them directly in the column's dtype property
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
            if _is_null_host_scalar(data):
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
        data_plc_col = data_col.to_pylibcudf(mode="read")
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
        return cls.from_pylibcudf(plc_column)

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

        with acquire_spill_lock():
            plc_column = plc.strings.convert.convert_lists.format_list_column(
                lc.to_pylibcudf(mode="read"),
                pa_scalar_to_plc_scalar(pa.scalar("None")),
                self._string_separators,
            )
            return type(self).from_pylibcudf(plc_column)  # type: ignore[return-value]

    def _transform_leaves(
        self, func: Callable[[ColumnBase, DtypeObj], ColumnBase], *args: Any
    ) -> Self:
        """
        Return a new column like Self but with func applied to the last leaf column.
        """
        leaf_queue: list[ListColumn] = []
        curr_col: ColumnBase = self

        while isinstance(curr_col, ListColumn):
            leaf_queue.append(curr_col)
            curr_col = curr_col.children[1]

        plc_leaf_col = func(curr_col, *args).to_pylibcudf(mode="read")

        # Rebuild the list column replacing just the leaf child
        while leaf_queue:
            col = leaf_queue.pop()
            offsets = col.children[0].to_pylibcudf(mode="read")
            plc_leaf_col = plc.Column(
                plc.DataType(plc.TypeId.LIST),
                col.size,
                None,
                plc.gpumemoryview(col.mask) if col.mask is not None else None,
                col.null_count,
                col.offset,
                [offsets, plc_leaf_col],
            )
        return type(self).from_pylibcudf(plc_leaf_col)

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
        if arrow_type or (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(self.dtype, pd.ArrowDtype)
        ):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        else:
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    @acquire_spill_lock()
    def count_elements(self) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.count_elements(self.to_pylibcudf(mode="read"))
        )

    @acquire_spill_lock()
    def distinct(self, nulls_equal: bool, nans_all_equal: bool) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.distinct(
                self.to_pylibcudf(mode="read"),
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
            )
        )

    @acquire_spill_lock()
    def sort_lists(
        self, ascending: bool, na_position: Literal["first", "last"]
    ) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.sort_lists(
                self.to_pylibcudf(mode="read"),
                plc.types.Order.ASCENDING
                if ascending
                else plc.types.Order.DESCENDING,
                (
                    plc.types.NullOrder.BEFORE
                    if na_position == "first"
                    else plc.types.NullOrder.AFTER
                ),
                False,
            )
        )

    @acquire_spill_lock()
    def extract_element_scalar(self, index: int) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.extract_list_element(
                self.to_pylibcudf(mode="read"),
                index,
            )
        )

    @acquire_spill_lock()
    def extract_element_column(self, index: ColumnBase) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.extract_list_element(
                self.to_pylibcudf(mode="read"),
                index.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def contains_scalar(self, search_key: pa.Scalar) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.contains(
                self.to_pylibcudf(mode="read"),
                pa_scalar_to_plc_scalar(search_key),
            )
        )

    @acquire_spill_lock()
    def index_of_scalar(self, search_key: pa.Scalar) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.index_of(
                self.to_pylibcudf(mode="read"),
                pa_scalar_to_plc_scalar(search_key),
                plc.lists.DuplicateFindOption.FIND_FIRST,
            )
        )

    @acquire_spill_lock()
    def index_of_column(self, search_keys: ColumnBase) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.index_of(
                self.to_pylibcudf(mode="read"),
                search_keys.to_pylibcudf(mode="read"),
                plc.lists.DuplicateFindOption.FIND_FIRST,
            )
        )

    @acquire_spill_lock()
    def concatenate_rows(self, other_columns: list[ColumnBase]) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.concatenate_rows(
                plc.Table(
                    [
                        col.to_pylibcudf(mode="read")
                        for col in itertools.chain([self], other_columns)
                    ]
                )
            )
        )

    @acquire_spill_lock()
    def concatenate_list_elements(self, dropna: bool) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.concatenate_list_elements(
                self.to_pylibcudf(mode="read"),
                plc.lists.ConcatenateNullPolicy.IGNORE
                if dropna
                else plc.lists.ConcatenateNullPolicy.NULLIFY_OUTPUT_ROW,
            )
        )

    @acquire_spill_lock()
    def segmented_gather(self, gather_map: ColumnBase) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.segmented_gather(
                self.to_pylibcudf(mode="read"),
                gather_map.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def join_list_elements(
        self,
        separator: str | StringColumn,
        sep_na_rep: str,
        string_na_rep: str,
    ) -> StringColumn:
        if isinstance(separator, str):
            sep: plc.Scalar | plc.Column = pa_scalar_to_plc_scalar(
                pa.scalar(separator)
            )
        else:
            sep = separator.to_pylibcudf(mode="read")
        plc_column = plc.strings.combine.join_list_elements(
            self.to_pylibcudf(mode="read"),
            sep,
            pa_scalar_to_plc_scalar(pa.scalar(sep_na_rep)),
            pa_scalar_to_plc_scalar(pa.scalar(string_na_rep)),
            plc.strings.combine.SeparatorOnNulls.YES,
            plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
        )
        return type(self).from_pylibcudf(plc_column)  # type: ignore[return-value]

    @acquire_spill_lock()
    def minhash_ngrams(
        self,
        width: int,
        seed: int | np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        # Convert int to np.uint32 with validation
        if isinstance(seed, int):
            if seed < 0 or seed > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                )
            seed = np.uint32(seed)
        return type(self).from_pylibcudf(
            plc.nvtext.minhash.minhash_ngrams(
                self.to_pylibcudf(mode="read"),
                width,
                seed,
                a.to_pylibcudf(mode="read"),
                b.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def minhash64_ngrams(
        self,
        width: int,
        seed: int | np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        # Convert int to np.uint64 with validation
        if isinstance(seed, int):
            if seed < 0 or seed > np.iinfo(np.uint64).max:
                raise ValueError(
                    f"seed must be in range [0, {np.iinfo(np.uint64).max}]"
                )
            seed = np.uint64(seed)
        return type(self).from_pylibcudf(
            plc.nvtext.minhash.minhash64_ngrams(
                self.to_pylibcudf(mode="read"),
                width,
                seed,
                a.to_pylibcudf(mode="read"),
                b.to_pylibcudf(mode="read"),
            )
        )
