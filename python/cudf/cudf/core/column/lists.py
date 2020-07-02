import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase
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

    def serialize(self):
        pass

    def deserialize(self, header, frames):
        pass

    @property
    def base_size(self):
        return self.size

    def get_children(self):
        return self._base_children

    @classmethod
    def from_arrow(cls, array):
        buffers = array.buffers()
        nlevels = len(buffers) / 2

        if nlevels == 1:
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
                children=(ListColumn.from_arrow(array.values), offsets),
            )

    def to_arrow(self):
        offsets = self.children[1].to_arrow()
        values = self.children[0]
        if self.nullable:
            nbuf = self.mask.to_host_array().view("int8")
            nbuf = pa.py_buffer(nbuf)
            buffers = (nbuf, offsets.buffers()[1])
        else:
            buffers = offsets.buffers()
        return pa.ListArray.from_buffers(
            self.dtype.to_arrow(),
            len(self),
            buffers,
            children=[values.to_arrow()],
        )
