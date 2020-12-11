import pyarrow as pa
import cudf
from cudf.core.column import StructColumn


class IntervalColumn(StructColumn):
    def __init__(
        self,
        dtype,
        mask=None,
        size=None,
        offset=0,
        null_count=None,
        children=(),
    ):

        super().__init__(
            data=None,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    @classmethod
    def from_arrow(self, data):
        new_col = super().from_arrow(data.storage)
        size = len(data)
        dtype = cudf.core.dtypes.IntervalDtype.from_arrow(data.type)
        mask = data.buffers()[0]
        if mask is not None:
            mask = cudf.utils.utils.pa_mask_buffer_to_mask(mask, len(data))

        offset = data.offset
        null_count = data.null_count
        children = new_col.children

        return IntervalColumn(
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def to_arrow(self):
        typ = self.dtype.to_arrow()
        return pa.ExtensionArray.from_storage(typ, super().to_arrow())

    def copy(self, deep=True):
        return super().copy(deep=deep).as_interval_column()
