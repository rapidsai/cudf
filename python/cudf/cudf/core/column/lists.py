# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

import cudf
from cudf._lib.copying import segmented_gather
from cudf._lib.lists import (
    concatenate_list_elements,
    concatenate_rows,
    contains_scalar,
    count_elements,
    distinct,
    extract_element_column,
    extract_element_scalar,
    index_of_column,
    index_of_scalar,
    sort_lists,
)
from cudf._lib.strings.convert.convert_lists import format_list_column
from cudf._typing import ColumnBinaryOperand, ColumnLike, Dtype, ScalarLike
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    is_list_dtype,
    is_scalar,
)
from cudf.core.column import ColumnBase, as_column, column
from cudf.core.column.methods import ColumnMethods, ParentType
from cudf.core.dtypes import ListDtype
from cudf.core.missing import NA


class ListColumn(ColumnBase):
    dtype: ListDtype
    _VALID_BINARY_OPERATIONS = {"__add__", "__radd__"}

    def __init__(
        self,
        size,
        dtype,
        mask=None,
        offset=0,
        null_count=None,
        children=(),
    ):
        super().__init__(
            None,
            size,
            dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    @cached_property
    def memory_usage(self):
        n = 0
        if self.nullable:
            n += cudf._lib.null_mask.bitmask_allocation_size_bytes(self.size)

        child0_size = (self.size + 1) * self.base_children[0].dtype.itemsize
        current_base_child = self.base_children[1]
        current_offset = self.offset
        n += child0_size
        while type(current_base_child) is ListColumn:
            child0_size = (
                current_base_child.size + 1 - current_offset
            ) * current_base_child.base_children[0].dtype.itemsize
            current_offset = current_base_child.base_children[
                0
            ].element_indexing(current_offset)
            n += child0_size
            current_base_child = current_base_child.base_children[1]

        n += (
            current_base_child.size - current_offset
        ) * current_base_child.dtype.itemsize

        if current_base_child.nullable:
            n += cudf._lib.null_mask.bitmask_allocation_size_bytes(
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
                return concatenate_rows([self, other])
            else:
                raise NotImplementedError(
                    "Lists concatenation for this operation is not yet"
                    "supported"
                )
        else:
            raise TypeError("can only concatenate list to list")

    @property
    def elements(self):
        """
        Column containing the elements of each list (may itself be a
        ListColumn)
        """
        return self.children[1]

    @property
    def offsets(self):
        """
        Integer offsets to elements specifying each row of the ListColumn
        """
        return self.children[0]

    def to_arrow(self):
        offsets = self.offsets.to_arrow()
        elements = (
            pa.nulls(len(self.elements))
            if len(self.elements) == self.elements.null_count
            else self.elements.to_arrow()
        )
        pa_type = pa.list_(elements.type)

        if self.nullable:
            nbuf = pa.py_buffer(self.mask.memoryview())
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

    def set_base_children(self, value: Tuple[ColumnBase, ...]):
        super().set_base_children(value)
        _, values = value
        self._dtype = cudf.ListDtype(element_type=values.dtype)

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Lists are not yet supported via `__cuda_array_interface__`"
        )

    def normalize_binop_value(self, other):
        if not isinstance(other, ListColumn):
            return NotImplemented
        return other

    def _with_type_metadata(
        self: "cudf.core.column.ListColumn", dtype: Dtype
    ) -> "cudf.core.column.ListColumn":
        if isinstance(dtype, ListDtype):
            return column.build_list_column(
                indices=self.base_children[0],
                elements=self.base_children[1]._with_type_metadata(
                    dtype.element_type
                ),
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
            )

        return self

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
        offset_col = [0]
        offset = 0

        # Build Data, Mask & Offsets
        for data in arbitrary:
            if cudf._lib.scalar._is_null_host_scalar(data):
                mask_col.append(False)
                offset_col.append(offset)
            else:
                mask_col.append(True)
                data_col = data_col.append(as_column(data))
                offset += len(data)
                offset_col.append(offset)

        offset_col = column.as_column(offset_col, dtype="int32")

        # Build ListColumn
        res = cls(
            size=len(arbitrary),
            dtype=cudf.ListDtype(data_col.dtype),
            mask=cudf._lib.transform.bools_to_mask(as_column(mask_col)),
            offset=0,
            null_count=0,
            children=(offset_col, data_col),
        )
        return res

    def as_string_column(
        self, dtype: Dtype, format=None, **kwargs
    ) -> "cudf.core.column.StringColumn":
        """
        Create a strings column from a list column
        """
        lc = self._transform_leaves(
            lambda col, dtype: col.as_string_column(dtype), dtype
        )

        # Separator strings to match the Python format
        separators = as_column([", ", "[", "]"])

        # Call libcudf to format the list column
        return format_list_column(lc, separators)

    def _transform_leaves(self, func, *args, **kwargs):
        # return a new list column with the same nested structure
        # as ``self``, but with the leaf column transformed
        # by applying ``func`` to it

        cc: List[ListColumn] = []
        c: ColumnBase = self

        while isinstance(c, ListColumn):
            cc.insert(0, c)
            c = c.children[1]

        lc = func(c, *args, **kwargs)

        # Rebuild the list column replacing just the leaf child
        for c in cc:
            o = c.children[0]
            lc = cudf.core.column.ListColumn(  # type: ignore
                size=c.size,
                dtype=cudf.ListDtype(lc.dtype),
                mask=c.mask,
                offset=c.offset,
                null_count=c.null_count,
                children=(o, lc),
            )
        return lc


class ListMethods(ColumnMethods):
    """
    List methods for Series
    """

    _column: ListColumn

    def __init__(self, parent: ParentType):
        if not is_list_dtype(parent.dtype):
            raise AttributeError(
                "Can only use .list accessor with a 'list' dtype"
            )
        super().__init__(parent=parent)

    def get(
        self,
        index: int,
        default: Optional[Union[ScalarLike, ColumnLike]] = None,
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
        if is_scalar(index):
            out = extract_element_scalar(self._column, cudf.Scalar(index))
        else:
            index = as_column(index)
            out = extract_element_column(self._column, as_column(index))

        if not (default is None or default is NA):
            # determine rows for which `index` is out-of-bounds
            lengths = count_elements(self._column)
            out_of_bounds_mask = (np.negative(index) > lengths) | (
                index >= lengths
            )

            # replace the value in those rows (should be NA) with `default`
            if out_of_bounds_mask.any():
                out = out._scatter_by_column(
                    out_of_bounds_mask, cudf.Scalar(default)
                )

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
        search_key = cudf.Scalar(search_key)
        try:
            res = self._return_or_inplace(
                contains_scalar(self._column, search_key)
            )
        except RuntimeError as e:
            if (
                "Type/Scale of search key does not "
                "match list column element type." in str(e)
            ):
                raise TypeError(str(e)) from e
            raise
        return res

    def index(self, search_key: Union[ScalarLike, ColumnLike]) -> ParentType:
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

        try:
            if is_scalar(search_key):
                return self._return_or_inplace(
                    index_of_scalar(self._column, cudf.Scalar(search_key))
                )
            else:
                return self._return_or_inplace(
                    index_of_column(self._column, as_column(search_key))
                )

        except RuntimeError as e:
            if (
                "Type/Scale of search key does not "
                "match list column element type." in str(e)
            ):
                raise TypeError(str(e)) from e
            raise

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
        return self._return_or_inplace(count_elements(self._column))

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
        if not _is_non_decimal_numeric_dtype(
            lists_indices_col.children[1].dtype
        ) or not np.issubdtype(
            lists_indices_col.children[1].dtype, np.integer
        ):
            raise TypeError(
                "lists_indices should be column of values of index types."
            )

        try:
            res = self._return_or_inplace(
                segmented_gather(self._column, lists_indices_col)
            )
        except RuntimeError as e:
            if "contains nulls" in str(e):
                raise ValueError("lists_indices contains null.") from e
            raise
        else:
            return res

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

        if is_list_dtype(self._column.children[1].dtype):
            raise NotImplementedError("Nested lists unique is not supported.")

        return self._return_or_inplace(
            distinct(self._column, nulls_equal=True, nans_all_equal=True)
        )

    def sort_values(
        self,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
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

        Notes
        -----
        Difference from pandas:
          * Not supporting: `inplace`, `kind`

        Examples
        --------
        >>> s = cudf.Series([[4, 2, None, 9], [8, 8, 2], [2, 1]])
        >>> s.list.sort_values(ascending=True, na_position="last")
        0    [2.0, 4.0, 9.0, nan]
        1         [2.0, 8.0, 8.0]
        2              [1.0, 2.0]
        dtype: list
        """
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if kind != "quicksort":
            raise NotImplementedError("`kind` not currently implemented.")
        if na_position not in {"first", "last"}:
            raise ValueError(f"Unknown `na_position` value {na_position}")
        if is_list_dtype(self._column.children[1].dtype):
            raise NotImplementedError("Nested lists sort is not supported.")

        return self._return_or_inplace(
            sort_lists(self._column, ascending, na_position),
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
        try:
            result = concatenate_list_elements(self._column, dropna=dropna)
        except RuntimeError as e:
            if "Rows of the input column must be lists." in str(e):
                raise ValueError(
                    "list.concat() can only be called on "
                    "list columns with at least one level "
                    "of nesting"
                )
        return self._return_or_inplace(result)

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
