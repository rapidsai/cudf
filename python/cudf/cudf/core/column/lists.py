import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase
from cudf.core.column.methods import ColumnMethodsMixin
from cudf.core.dtypes import ListDtype
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

    def get_children(self):
        return self._base_children

    @property
    def values(self):
        return self.children[0]

    @property
    def indices(self):
        return self.children[1]

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
                children=(ListColumn.from_arrow(array.values), offsets),
            )

    def to_arrow(self):
        offsets = self.children[1].to_arrow()
        values = self.children[0].to_arrow()
        if len(values) == values.null_count:
            values = pa.NullArray.from_pandas([None] * len(values))
        if self.nullable:
            nbuf = self.mask.to_host_array().view("int8")
            nbuf = pa.py_buffer(nbuf)
            buffers = (nbuf, offsets.buffers()[1])
        else:
            buffers = offsets.buffers()
        return pa.ListArray.from_buffers(
            self.dtype.to_arrow(), len(self), buffers, children=[values],
        )

    def list(self, parent=None):
        return ListMethods(self, parent=parent)


class ListMethods(ColumnMethodsMixin):
    """
    List methods for Series
    """

    def __init__(self, column, parent=None):
        self._column = column
        self._parent = parent

    @property
    def leaves(self):
        """
        From a Series of (possibly nested) lists, obtain the values from
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
        if type(self._column.values) is ListColumn:
            return self._column.values.list(parent=self._parent).leaves
        else:
            return self._return_or_inplace(
                self._column.values, retain_index=False
            )
