# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import pickle

import numpy as np
import pyarrow as pa

import cudf
from cudf._lib.copying import segmented_gather
from cudf._lib.lists import (
    contains_scalar,
    count_elements,
    drop_list_duplicates,
    extract_element,
    sort_lists,
)
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase, as_column, column
from cudf.core.column.methods import ColumnMethodsMixin
from cudf.utils.dtypes import is_list_dtype, is_numerical_dtype


class ListColumn(ColumnBase):
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

    @property
    def base_size(self):
        return len(self.base_children[0]) - 1

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
        header["type-serialized"] = pickle.dumps(type(self))
        header["dtype"] = pickle.dumps(self.dtype)
        header["null_count"] = self.null_count
        header["size"] = self.size

        frames = []
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

        # Deserialize child columns
        children = []
        f = 0
        for h in header["subheaders"]:
            fcount = h["frame_count"]
            child_frames = frames[f : f + fcount]
            column_type = pickle.loads(h["type-serialized"])
            children.append(column_type.deserialize(h, child_frames))
            f += fcount

        # Materialize list column
        return column.build_column(
            data=None,
            dtype=pickle.loads(header["dtype"]),
            mask=mask,
            children=tuple(children),
            size=header["size"],
        )


class ListMethods(ColumnMethodsMixin):
    """
    List methods for Series
    """

    def __init__(self, column, parent=None):
        if not is_list_dtype(column.dtype):
            raise AttributeError(
                "Can only use .list accessor with a 'list' dtype"
            )
        super().__init__(column=column, parent=parent)

    def get(self, index):
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

    def contains(self, search_key):
        """
        Creates a column of bool values indicating whether the specified scalar
        is an element of each row of a list column.

        Parameters
        ----------
        search_key : scalar
            element being searched for in each row of the list column

        Returns
        -------
        Column

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
    def leaves(self):
        """
        From a Series of (possibly nested) lists, obtain the elements from
        the innermost lists as a flat Series (one value per row).

        Returns
        -------
        Series

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

    def len(self):
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

    def take(self, lists_indices):
        """
        Collect list elements based on given indices.

        Parameters
        ----------
        lists_indices: List type arrays
            Specifies what to collect from each row

        Returns
        -------
        ListColumn

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
        if not is_numerical_dtype(
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

    def unique(self):
        """
        Returns unique element for each list in the column, order for each
        unique element is not guaranteed.

        Returns
        -------
        ListColumn

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
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
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
        ListColumn with each list sorted

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
