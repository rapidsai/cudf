import numpy as np
from . import columnops, _gdf
from .buffer import Buffer
from .cudautils import compact_mask_bytes


class DatetimeColumn(columnops.TypedColumnBase):
    def __init__(self, data, mask=None, null_count=None, dtype=None):
        # currently libgdf datetime kernels fail if mask is null
        if mask is None:
            mask = np.ones(data.mem.size, dtype=np.bool)
            mask = compact_mask_bytes(mask)
            mask = Buffer(mask)
        super(DatetimeColumn, self).__init__(data=data,
                                             mask=mask,
                                             null_count=null_count,
                                             dtype=dtype
                                             )
        # the column constructor removes mask if it's all true
        self._mask = mask

    @classmethod
    def from_numpy(cls, array):
        # hack, coerce to int, then set the dtype
        array = array.astype('datetime64[ms]')
        dtype = np.int64
        assert array.dtype.itemsize == 8
        buf = Buffer(array.astype(dtype, copy=False))
        buf.dtype = array.dtype
        return cls(data=buf, dtype=buf.dtype)


def extract_dt_field(op, input_column):
    out = columnops.column_empty_like_same_mask(
        input_column,
        dtype=np.int16
    )
    # force mask again
    out._mask = input_column.mask
    _gdf.apply_unaryop(op,
                       input_column,
                       out)
    return out
