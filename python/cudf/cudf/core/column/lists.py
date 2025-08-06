# Copyright (c) 2020-2025, NVIDIA CORPORATION.

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
    SIZE_TYPE_DTYPE,
    get_dtype_of_same_kind,
    is_dtype_obj_list,
)
from cudf.utils.scalar import (
    maybe_nested_pa_scalar_to_py,
    pa_scalar_to_plc_scalar,
)
from cudf.utils.utils import _is_null_host_scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from cudf._typing import ColumnBinaryOperand, ColumnLike, Dtype
    from cudf.core.buffer import Buffer
    from cudf.core.column.string import StringColumn


class ListColumn(ColumnBase):
    _VALID_BINARY_OPERATIONS = {"__add__", "__radd__"}

    def __init__(
        self,
        data: None,
        size: int,
        dtype: ListDtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple[NumericalColumn, ColumnBase] = (),  # type: ignore[assignment]
    ):
        if data is not None:
            raise ValueError("data must be None")
        if (
            not cudf.get_option("mode.pandas_compatible")
            and not isinstance(dtype, ListDtype)
        ) or (
            cudf.get_option("mode.pandas_compatible")
            and not is_dtype_obj_list(dtype)
        ):
            raise ValueError("dtype must be a cudf.ListDtype")
        if not (
            len(children) == 2
            and isinstance(children[0], NumericalColumn)
            # TODO: Enforce int32_t (size_type) used in libcudf?
            and children[0].dtype.kind == "i"
            and isinstance(children[1], ColumnBase)
        ):
            raise ValueError(
                "children must a tuple of 2 columns of (signed integer offsets, list values)"
            )
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
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
            return self.dtype._recursively_replace_fields(py_element)
        return result

    def _cast_setitem_value(self, value: Any) -> plc.Scalar:
        if isinstance(value, list) or value is None:
            return pa_scalar_to_plc_scalar(
                pa.scalar(value, type=self.dtype.to_arrow())
            )
        elif value is NA or value is None:
            return pa_scalar_to_plc_scalar(
                pa.scalar(None, type=self.dtype.to_arrow())
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
        other = self._normalize_binop_operand(other)
        if other is NotImplemented:
            return NotImplemented
        if isinstance(other.dtype, ListDtype):
            if op == "__add__":
                return self.concatenate_rows([other])  # type: ignore[list-item]
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
            buffers = (nbuf, offsets.buffers()[1])
        else:
            buffers = offsets.buffers()
        return pa.ListArray.from_buffers(
            pa_type, len(self), buffers, children=[elements]
        )

    def set_base_data(self, value):
        if value is not None:
            raise RuntimeError(
                "ListColumn's do not use data attribute of Column, use "
                "`set_base_children` instead"
            )
        else:
            super().set_base_data(value)

    def set_base_children(self, value: tuple[NumericalColumn, ColumnBase]):  # type: ignore[override]
        super().set_base_children(value)
        self._dtype = cudf.ListDtype(element_type=value[1].dtype)

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Lists are not yet supported via `__cuda_array_interface__`"
        )

    def _normalize_binop_operand(self, other: Any) -> ColumnBase:
        if isinstance(other, type(self)):
            return other
        return NotImplemented

    def _with_type_metadata(self: Self, dtype: Dtype) -> Self:
        if isinstance(dtype, ListDtype):
            elements = self.base_children[1]._with_type_metadata(
                dtype.element_type
            )
            return type(self)(
                data=None,
                dtype=dtype,
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
                children=(self.base_children[0], elements),  # type: ignore[arg-type]
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

    def leaves(self):
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
        mask_col = []
        offset_vals = [0]
        offset = 0

        # Build Data, Mask & Offsets
        for data in arbitrary:
            if _is_null_host_scalar(data):
                mask_col.append(False)
                offset_vals.append(offset)
            else:
                mask_col.append(True)
                data_col = data_col.append(as_column(data))
                offset += len(data)
                offset_vals.append(offset)

        offset_col = cast(
            NumericalColumn,
            as_column(offset_vals, dtype=SIZE_TYPE_DTYPE),
        )

        # Build ListColumn
        res = cls(
            data=None,
            size=len(arbitrary),
            dtype=cudf.ListDtype(data_col.dtype),
            mask=as_column(mask_col).as_mask(),
            offset=0,
            null_count=0,
            children=(offset_col, data_col),
        )
        return res

    def as_string_column(self, dtype) -> StringColumn:
        """
        Create a strings column from a list column
        """
        if cudf.get_option("mode.pandas_compatible"):
            if isinstance(dtype, np.dtype) and dtype.kind == "O":
                raise TypeError(
                    f"Cannot cast a list from {self.dtype} to {dtype}"
                )
        lc = self._transform_leaves(lambda col: col.as_string_column(dtype))

        # Separator strings to match the Python format
        separators = as_column([", ", "[", "]"])

        with acquire_spill_lock():
            plc_column = plc.strings.convert.convert_lists.format_list_column(
                lc.to_pylibcudf(mode="read"),
                pa_scalar_to_plc_scalar(pa.scalar("None")),
                separators.to_pylibcudf(mode="read"),
            )
            return type(self).from_pylibcudf(plc_column)  # type: ignore[return-value]

    def _transform_leaves(self, func, *args, **kwargs) -> Self:
        # return a new list column with the same nested structure
        # as ``self``, but with the leaf column transformed
        # by applying ``func`` to it

        cc: list[ListColumn] = []
        c: ColumnBase = self

        while isinstance(c, ListColumn):
            cc.insert(0, c)
            c = c.children[1]

        lc = func(c, *args, **kwargs)

        # Rebuild the list column replacing just the leaf child
        for c in cc:
            o = c.children[0]
            lc = ListColumn(  # type: ignore
                data=None,
                size=c.size,
                dtype=cudf.ListDtype(lc.dtype),
                mask=c.mask,
                offset=c.offset,
                null_count=c.null_count,
                children=(o, lc),  # type: ignore[arg-type]
            )
        return lc

    @property
    def element_type(self) -> Dtype:
        """
        Returns the element type of the list column.
        """
        if isinstance(self.dtype, ListDtype):
            return self.dtype.element_type
        else:
            return get_dtype_of_same_kind(
                self.dtype,
                self.dtype.pyarrow_dtype.value_type.to_pandas_dtype(),
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
            sep = pa_scalar_to_plc_scalar(pa.scalar(separator))
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
        seed: np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
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
        seed: np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.minhash.minhash64_ngrams(
                self.to_pylibcudf(mode="read"),
                width,
                seed,
                a.to_pylibcudf(mode="read"),
                b.to_pylibcudf(mode="read"),
            )
        )
