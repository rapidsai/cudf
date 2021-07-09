# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import pickle

import numpy as np
import pyarrow as pa

import cudf
from cudf._lib.copying import segmented_gather
from cudf._lib.lists import (
    concatenate_list_elements,
    concatenate_rows,
    contains_scalar,
    count_elements,
    drop_list_duplicates,
    extract_element,
    sort_lists,
)
from cudf._lib.table import Table
from cudf._typing import BinaryOperand, ColumnLike, Dtype, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase, as_column, column
from cudf.core.column.methods import ColumnMethodsMixin, ParentType
from cudf.core.dtypes import ListDtype
from cudf.utils.dtypes import _is_non_decimal_numeric_dtype, is_list_dtype


class ListColumn(ColumnBase):
    dtype: ListDtype

    def __init__(
        self, size, dtype, mask=None, offset=0, null_count=None, children=(),
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

    def __sizeof__(self):
        if self._cached_sizeof is None:
            n = 0
            if self.nullable:
                n += cudf._lib.null_mask.bitmask_allocation_size_bytes(
                    self.size
                )

            child0_size = (self.size + 1) * self.base_children[
                0
            ].dtype.itemsize
            current_base_child = self.base_children[1]
            current_offset = self.offset
            n += child0_size
            while type(current_base_child) is ListColumn:
                child0_size = (
                    current_base_child.size + 1 - current_offset
                ) * current_base_child.base_children[0].dtype.itemsize
                current_offset = current_base_child.base_children[0][
                    current_offset
                ]
                n += child0_size
                current_base_child = current_base_child.base_children[1]

            n += (
                current_base_child.size - current_offset
            ) * current_base_child.dtype.itemsize

            if current_base_child.nullable:
                n += cudf._lib.null_mask.bitmask_allocation_size_bytes(
                    current_base_child.size
                )
            self._cached_sizeof = n

        return self._cached_sizeof

    def __setitem__(self, key, value):
        if isinstance(value, list):
            value = cudf.Scalar(value)
        if isinstance(value, cudf.Scalar):
            if value.dtype != self.dtype:
                raise TypeError("list nesting level mismatch")
        elif value is cudf.NA:
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

    def binary_operator(
        self, binop: str, other: BinaryOperand, reflect: bool = False
    ) -> ColumnBase:
        """
        Calls a binary operator *binop* on operands *self*
        and *other*.

        Parameters
        ----------
        self, other : list columns

        binop :  binary operator
            Only "add" operator is currently being supported
            for lists concatenation functions

        reflect : boolean, default False
            If ``reflect`` is ``True``, swap the order of
            the operands.

        Returns
        -------
        Series : the output dtype is determined by the
            input operands.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({'val': [['a', 'a'], ['b'], ['c']]})
        >>> gdf
            val
        0  [a, a]
        1     [b]
        2     [c]
        >>> gdf['val'] + gdf['val']
        0    [a, a, a, a]
        1          [b, b]
        2          [c, c]
        Name: val, dtype: list

        """

        if isinstance(other.dtype, ListDtype):
            if binop == "add":
                return concatenate_rows(Table({0: self, 1: other}))
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

    def list(self, parent=None):
        return ListMethods(self, parent=parent)

    def to_arrow(self):
        offsets = self.offsets.to_arrow()
        elements = (
            pa.nulls(len(self.elements))
            if len(self.elements) == self.elements.null_count
            else self.elements.to_arrow()
        )
        pa_type = pa.list_(elements.type)

        if self.nullable:
            nbuf = self.mask.to_host_array().view("int8")
            nbuf = pa.py_buffer(nbuf)
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

    def serialize(self):
        header = {}
        frames = []
        header["type-serialized"] = pickle.dumps(type(self))
        header["null_count"] = self.null_count
        header["size"] = self.size
        header["dtype"], dtype_frames = self.dtype.serialize()
        header["dtype_frames_count"] = len(dtype_frames)
        frames.extend(dtype_frames)

        sub_headers = []

        for item in self.children:
            sheader, sframes = item.serialize()
            sub_headers.append(sheader)
            frames.extend(sframes)

        if self.null_count > 0:
            frames.append(self.mask)

        header["subheaders"] = sub_headers
        header["frame_count"] = len(frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):

        # Get null mask
        if header["null_count"] > 0:
            mask = Buffer(frames[-1])
        else:
            mask = None

        # Deserialize dtype
        dtype = pickle.loads(header["dtype"]["type-serialized"]).deserialize(
            header["dtype"], frames[: header["dtype_frames_count"]]
        )

        # Deserialize child columns
        children = []
        f = header["dtype_frames_count"]
        for h in header["subheaders"]:
            fcount = h["frame_count"]
            child_frames = frames[f : f + fcount]
            column_type = pickle.loads(h["type-serialized"])
            children.append(column_type.deserialize(h, child_frames))
            f += fcount

        # Materialize list column
        return column.build_column(
            data=None,
            dtype=dtype,
            mask=mask,
            children=tuple(children),
            size=header["size"],
        )

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Lists are not yet supported via `__cuda_array_interface__`"
        )

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


class ListMethods(ColumnMethodsMixin):
    """
    List methods for Series
    """

    _column: ListColumn

    def __init__(self, column: ListColumn, parent: ParentType = None):
        if not is_list_dtype(column.dtype):
            raise AttributeError(
                "Can only use .list accessor with a 'list' dtype"
            )
        super().__init__(column=column, parent=parent)

    def get(self, index: int) -> ParentType:
        """
        Extract element at the given index from each component

        Extract element from lists, tuples, or strings in
        each element in the Series/Index.

        Parameters
        ----------
        index : int

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
        """
        min_col_list_len = self.len().min()
        if -min_col_list_len <= index < min_col_list_len:
            return self._return_or_inplace(
                extract_element(self._column, index)
            )
        else:
            raise IndexError("list index out of range")

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
                "Type/Scale of search key does not"
                "match list column element type" in str(e)
            ):
                raise TypeError(
                    "Type/Scale of search key does not"
                    "match list column element type"
                ) from e
            raise
        else:
            return res

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
        if type(self._column.elements) is ListColumn:
            return self._column.elements.list(parent=self._parent).leaves
        else:
            return self._return_or_inplace(
                self._column.elements, retain_index=False
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
            drop_list_duplicates(
                self._column, nulls_equal=True, nans_all_equal=True
            )
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
