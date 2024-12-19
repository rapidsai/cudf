# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
import pyarrow as pa
from typing_extensions import Self

import pylibcudf as plc

import cudf
import cudf.core.column.column as column
from cudf._lib.types import size_type_dtype
from cudf.api.types import _is_non_decimal_numeric_dtype, is_scalar
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.column.methods import ColumnMethods, ParentType
from cudf.core.column.numerical import NumericalColumn
from cudf.core.dtypes import ListDtype
from cudf.core.missing import NA

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf._typing import ColumnBinaryOperand, ColumnLike, Dtype, ScalarLike
    from cudf.core.buffer import Buffer


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
        if not isinstance(dtype, ListDtype):
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

    @cached_property
    def memory_usage(self):
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

    def __setitem__(self, key, value):
        if isinstance(value, list):
            value = cudf.Scalar(value)
        if isinstance(value, cudf.Scalar):
            if value.dtype != self.dtype:
                raise TypeError("list nesting level mismatch")
        elif value is NA:
            value = cudf.Scalar(value, dtype=self.dtype)
        else:
            raise ValueError(f"Can not set {value} into ListColumn")
        super().__setitem__(key, value)

    @property
    def base_size(self):
        # in some cases, libcudf will return an empty ListColumn with no
        # indices; in these cases, we must manually set the base_size to 0 to
        # avoid it being negative
        return max(0, len(self.base_children[0]) - 1)

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        # Lists only support __add__, which concatenates lists.
        reflect, op = self._check_reflected_op(op)
        other = self._wrap_binop_normalization(other)
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

    def normalize_binop_value(self, other) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        return other

    def _with_type_metadata(
        self: "cudf.core.column.ListColumn", dtype: Dtype
    ) -> "cudf.core.column.ListColumn":
        if isinstance(dtype, ListDtype):
            elements = self.base_children[1]._with_type_metadata(
                dtype.element_type
            )
            return ListColumn(
                data=None,
                dtype=dtype,
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
                children=(self.base_children[0], elements),  # type: ignore[arg-type]
            )

        return self

    def copy(self, deep: bool = True):
        # Since list columns are immutable, both deep and shallow copies share
        # the underlying device data and mask.
        return super().copy(deep=False)

    def leaves(self):
        if isinstance(self.elements, ListColumn):
            return self.elements.leaves()
        else:
            return self.elements

    @classmethod
    def from_sequences(
        cls, arbitrary: Sequence[ColumnLike]
    ) -> "cudf.core.column.ListColumn":
        """
        Create a list column for list of column-like sequences
        """
        data_col = column.column_empty(0)
        mask_col = []
        offset_vals = [0]
        offset = 0

        # Build Data, Mask & Offsets
        for data in arbitrary:
            if cudf._lib.scalar._is_null_host_scalar(data):
                mask_col.append(False)
                offset_vals.append(offset)
            else:
                mask_col.append(True)
                data_col = data_col.append(as_column(data))
                offset += len(data)
                offset_vals.append(offset)

        offset_col = cast(
            NumericalColumn,
            column.as_column(offset_vals, dtype=size_type_dtype),
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

    def as_string_column(self) -> cudf.core.column.StringColumn:
        """
        Create a strings column from a list column
        """
        lc = self._transform_leaves(lambda col: col.as_string_column())

        # Separator strings to match the Python format
        separators = as_column([", ", "[", "]"])

        with acquire_spill_lock():
            plc_column = plc.strings.convert.convert_lists.format_list_column(
                lc.to_pylibcudf(mode="read"),
                cudf.Scalar("None").device_value.c_value,
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
            lc = cudf.core.column.ListColumn(  # type: ignore
                data=None,
                size=c.size,
                dtype=cudf.ListDtype(lc.dtype),
                mask=c.mask,
                offset=c.offset,
                null_count=c.null_count,
                children=(o, lc),  # type: ignore[arg-type]
            )
        return lc

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or nullable:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
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
    def contains_scalar(self, search_key: cudf.Scalar) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.contains(
                self.to_pylibcudf(mode="read"),
                search_key.device_value.c_value,
            )
        )

    @acquire_spill_lock()
    def index_of_scalar(self, search_key: cudf.Scalar) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.lists.index_of(
                self.to_pylibcudf(mode="read"),
                search_key.device_value.c_value,
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


class ListMethods(ColumnMethods):
    """
    List methods for Series
    """

    _column: ListColumn

    def __init__(self, parent: ParentType):
        if not isinstance(parent.dtype, ListDtype):
            raise AttributeError(
                "Can only use .list accessor with a 'list' dtype"
            )
        super().__init__(parent=parent)

    def get(
        self,
        index: int | ColumnLike,
        default: ScalarLike | ColumnLike | None = None,
    ) -> ParentType:
        """
        Extract element at the given index from each list in a Series of lists.

        ``index`` can be an integer or a sequence of integers.  If
        ``index`` is an integer, the element at position ``index`` is
        extracted from each list.  If ``index`` is a sequence, it must
        be of the same length as the Series, and ``index[i]``
        specifies the position of the element to extract from the
        ``i``-th list in the Series.

        If the index is out of bounds for any list, return <NA> or, if
        provided, ``default``.  Thus, this method never raises an
        ``IndexError``.

        Parameters
        ----------
        index : int or sequence of ints
        default : scalar, optional

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> s.list.get(-1)
        0    3
        1    5
        2    6
        dtype: int64

        >>> s = cudf.Series([[1, 2], [3, 4, 5], [4, 5, 6]])
        >>> s.list.get(2)
        0    <NA>
        1       5
        2       6
        dtype: int64

        >>> s.list.get(2, default=0)
        0   0
        1   5
        2   6
        dtype: int64

        >>> s.list.get([0, 1, 2])
        0   1
        1   4
        2   6
        dtype: int64
        """
        if isinstance(index, int):
            out = self._column.extract_element_scalar(index)
        else:
            index = as_column(index)
            out = self._column.extract_element_column(index)

        if not (default is None or default is NA):
            # determine rows for which `index` is out-of-bounds
            lengths = self._column.count_elements()
            out_of_bounds_mask = ((-1 * index) > lengths) | (index >= lengths)

            # replace the value in those rows (should be NA) with `default`
            if out_of_bounds_mask.any():
                out = out._scatter_by_column(
                    out_of_bounds_mask, cudf.Scalar(default)
                )
        if out.dtype != self._column.dtype.element_type:
            # libcudf doesn't maintain struct labels so we must transfer over
            # manually from the input column if we lost some information
            # somewhere. Not doing this unilaterally since the cost is
            # non-zero..
            out = out._with_type_metadata(self._column.dtype.element_type)
        return self._return_or_inplace(out)

    def contains(self, search_key: ScalarLike) -> ParentType:
        """
        Returns boolean values indicating whether the specified scalar
        is an element of each row.

        Parameters
        ----------
        search_key : scalar
            element being searched for in each row of the list column

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> s.list.contains(4)
        Series([False, True, True])
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.contains_scalar(cudf.Scalar(search_key))
        )

    def index(self, search_key: ScalarLike | ColumnLike) -> ParentType:
        """
        Returns integers representing the index of the search key for each row.

        If ``search_key`` is a sequence, it must be the same length as the
        Series and ``search_key[i]`` represents the search key for the
        ``i``-th row of the Series.

        If the search key is not contained in a row, -1 is returned. If either
        the row or the search key are null, <NA> is returned. If the search key
        is contained multiple times, the smallest matching index is returned.

        Parameters
        ----------
        search_key : scalar or sequence of scalars
            Element or elements being searched for in each row of the list
            column

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> s.list.index(4)
        0   -1
        1    1
        2    0
        dtype: int32

        >>> s = cudf.Series([["a", "b", "c"], ["x", "y", "z"]])
        >>> s.list.index(["b", "z"])
        0    1
        1    2
        dtype: int32

        >>> s = cudf.Series([[4, 5, 6], None, [-3, -2, -1]])
        >>> s.list.index([None, 3, -2])
        0    <NA>
        1    <NA>
        2       1
        dtype: int32
        """

        if is_scalar(search_key):
            result = self._column.index_of_scalar(cudf.Scalar(search_key))
        else:
            result = self._column.index_of_column(as_column(search_key))
        return self._return_or_inplace(result)

    @property
    def leaves(self) -> ParentType:
        """
        From a Series of (possibly nested) lists, obtain the elements from
        the innermost lists as a flat Series (one value per row).

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> a = cudf.Series([[[1, None], [3, 4]], None, [[5, 6]]])
        >>> a.list.leaves
        0       1
        1    <NA>
        2       3
        3       4
        4       5
        5       6
        dtype: int64
        """
        return self._return_or_inplace(
            self._column.leaves(), retain_index=False
        )

    def len(self) -> ParentType:
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> s
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> s.list.len()
        0       3
        1    <NA>
        2       2
        dtype: int32
        """
        return self._return_or_inplace(self._column.count_elements())

    def take(self, lists_indices: ColumnLike) -> ParentType:
        """
        Collect list elements based on given indices.

        Parameters
        ----------
        lists_indices: Series-like of lists
            Specifies what to collect from each row

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> s
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> s.list.take([[0, 1], [], []])
        0    [1, 2]
        1      None
        2        []
        dtype: list
        """

        lists_indices_col = as_column(lists_indices)
        if not isinstance(lists_indices_col, ListColumn):
            raise ValueError("lists_indices should be list type array.")
        if not lists_indices_col.size == self._column.size:
            raise ValueError(
                "lists_indices and list column is of different " "size."
            )
        if (
            not _is_non_decimal_numeric_dtype(
                lists_indices_col.children[1].dtype
            )
            or lists_indices_col.children[1].dtype.kind not in "iu"
        ):
            raise TypeError(
                "lists_indices should be column of values of index types."
            )

        return self._return_or_inplace(
            self._column.segmented_gather(lists_indices_col)
        )

    def unique(self) -> ParentType:
        """
        Returns the unique elements in each list.
        The ordering of elements is not guaranteed.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 1, 2, None, None], None, [4, 4], []])
        >>> s
        0    [1.0, 1.0, 2.0, nan, nan]
        1                         None
        2                   [4.0, 4.0]
        3                           []
        dtype: list
        >>> s.list.unique() # Order of list element is not guaranteed
        0              [1.0, 2.0, nan]
        1                         None
        2                        [4.0]
        3                           []
        dtype: list
        """

        if isinstance(self._column.children[1].dtype, ListDtype):
            raise NotImplementedError("Nested lists unique is not supported.")

        return self._return_or_inplace(
            self._column.distinct(nulls_equal=True, nans_all_equal=True)
        )

    def sort_values(
        self,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: Literal["first", "last"] = "last",
        ignore_index: bool = False,
    ) -> ParentType:
        """
        Sort each list by the values.

        Sort the lists in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {'first', 'last'}, default 'last'
            'first' puts nulls at the beginning, 'last' puts nulls at the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.

        Returns
        -------
        Series or Index with each list sorted

        Examples
        --------
        >>> s = cudf.Series([[4, 2, None, 9], [8, 8, 2], [2, 1]])
        >>> s.list.sort_values(ascending=True, na_position="last")
        0    [2.0, 4.0, 9.0, nan]
        1         [2.0, 8.0, 8.0]
        2              [1.0, 2.0]
        dtype: list

        .. pandas-compat::
            `pandas.Series.list.sort_values`

            This method does not exist in pandas but it can be run
            as:

            >>> import pandas as pd
            >>> s = pd.Series([[3, 2, 1], [2, 4, 3]])
            >>> print(s.apply(sorted))
            0    [1, 2, 3]
            1    [2, 3, 4]
            dtype: object
        """
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if kind != "quicksort":
            raise NotImplementedError("`kind` not currently implemented.")
        if na_position not in {"first", "last"}:
            raise ValueError(f"Unknown `na_position` value {na_position}")
        if isinstance(self._column.children[1].dtype, ListDtype):
            raise NotImplementedError("Nested lists sort is not supported.")

        return self._return_or_inplace(
            self._column.sort_lists(ascending, na_position),
            retain_index=not ignore_index,
        )

    def concat(self, dropna=True) -> ParentType:
        """
        For a column with at least one level of nesting, concatenate the
        lists in each row.

        Parameters
        ----------
        dropna: bool, optional
            If True (default), ignores top-level null elements in each row.
            If False, and top-level null elements are present, the resulting
            row in the output is null.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s1
        0      [[1.0, 2.0], [3.0, 4.0, 5.0]]
        1    [[6.0, None], [7.0], [8.0, 9.0]]
        dtype: list
        >>> s1.list.concat()
        0    [1.0, 2.0, 3.0, 4.0, 5.0]
        1    [6.0, None, 7.0, 8.0, 9.0]
        dtype: list

        Null values at the top-level in each row are dropped by default:

        >>> s2
        0    [[1.0, 2.0], None, [3.0, 4.0, 5.0]]
        1        [[6.0, None], [7.0], [8.0, 9.0]]
        dtype: list
        >>> s2.list.concat()
        0    [1.0, 2.0, 3.0, 4.0, 5.0]
        1    [6.0, None, 7.0, 8.0, 9.0]
        dtype: list

        Use ``dropna=False`` to produce a null instead:

        >>> s2.list.concat(dropna=False)
        0                         None
        1    [6.0, nan, 7.0, 8.0, 9.0]
        dtype: list
        """
        return self._return_or_inplace(
            self._column.concatenate_list_elements(dropna)
        )

    def astype(self, dtype):
        """
        Return a new list Series with the leaf values casted
        to the specified data type.

        Parameters
        ----------
        dtype: data type to cast leaves values to

        Returns
        -------
        A new Series of lists

        Examples
        --------
        >>> s = cudf.Series([[1, 2], [3, 4]])
        >>> s.dtype
        ListDtype(int64)
        >>> s2 = s.list.astype("float64")
        >>> s2.dtype
        ListDtype(float64)
        """
        return self._return_or_inplace(
            self._column._transform_leaves(
                lambda col, dtype: col.astype(dtype), dtype
            )
        )
