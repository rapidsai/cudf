# Copyright (c) 2020, NVIDIA CORPORATION.
import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase
from cudf.core.column.methods import ColumnMethodsMixin
from cudf.core.dtypes import ListDtype
from cudf.utils.dtypes import is_list_dtype
from cudf.utils.utils import buffers_from_pyarrow


class ListColumn(ColumnBase):
    def __init__(
        self,
        data,
        size,
        dtype,
        mask=None,
        offset=0,
        null_count=None,
        children=(),
    ):
        super().__init__(
            data,
            size,
            dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    @property
    def base_size(self):
        return self.size

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

    @classmethod
    def from_arrow(cls, array):
        if array.type.num_children == 0:
            return cudf.core.column.as_column(array)
        else:
            _, _, mask, _, _ = buffers_from_pyarrow(array)
            offsets = cudf.core.column.as_column(array.offsets)
            return ListColumn(
                data=None,
                size=len(array),
                dtype=ListDtype.from_arrow(array.type),
                offset=array.offset,
                mask=mask,
                null_count=array.null_count,
                children=(offsets, ListColumn.from_arrow(array.values)),
            )

    def to_arrow(self):
        offsets = self.offsets.to_arrow()
        elements = self.elements.to_arrow()
        if len(elements) == elements.null_count:
            elements = pa.NullArray.from_pandas([None] * len(elements))
        if self.nullable:
            nbuf = self.mask.to_host_array().view("int8")
            nbuf = pa.py_buffer(nbuf)
            buffers = (nbuf, offsets.buffers()[1])
        else:
            buffers = offsets.buffers()
        return pa.ListArray.from_buffers(
            self.dtype.to_arrow(), len(self), buffers, children=[elements],
        )

    def list(self, parent=None):
        return ListMethods(self, parent=parent)


class ListMethods(ColumnMethodsMixin):
    """
    List methods for Series
    """

    def __init__(self, column, parent=None):
        if not is_list_dtype(column.dtype):
            raise AttributeError(
                "Can only use .cat accessor with a 'list' dtype"
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
