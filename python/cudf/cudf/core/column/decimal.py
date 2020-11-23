from cudf.core.column import ColumnBase
from cudf.core.dtypes import DecimalDtype
from cudf.core.buffer import Buffer
import pyarrow as pa
import more_itertools


class DecimalColumn(ColumnBase):
    @classmethod
    def from_arrow(cls, data: pa.Array):
        bts = data.buffers()[1].to_pybytes()
        bts = b"".join(list(more_itertools.sliced(bts, 8))[::2])
        return cls(
            data=Buffer.from_bytes(bts),
            size=len(data),
            dtype=DecimalDtype.from_arrow(data.type),
        )
