# Copyright (c) 2020, NVIDIA CORPORATION.

import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase
from cudf.core.column.methods import ColumnMethodsMixin
from cudf.utils.dtypes import is_list_dtype


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


class ListMethods(ColumnMethodsMixin):
    """
    List methods for Series
    """

    def __init__(self, column, parent=None):
        if not is_list_dtype(column.dtype):
            raise AttributeError(
                "Can only use .list accessor with a 'list' dtype"
            )
        self._column = column
        self._parent = parent

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
        1    null
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
