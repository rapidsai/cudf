import numpy as np
from . import columnops, _gdf
from .buffer import Buffer
from .cudautils import compact_mask_bytes
from libgdf_cffi import libgdf


class DatetimeColumn(columnops.TypedColumnBase):
    funcs = {
        'year': libgdf.gdf_extract_datetime_year,
        'month': libgdf.gdf_extract_datetime_month,
        'day': libgdf.gdf_extract_datetime_day,
        'hour': libgdf.gdf_extract_datetime_hour,
        'minute': libgdf.gdf_extract_datetime_minute,
        'second': libgdf.gdf_extract_datetime_second,
    }

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

    @property
    def year(self):
        return self.get_dt_field('year')

    @property
    def month(self):
        return self.get_dt_field('month')

    @property
    def day(self):
        return self.get_dt_field('day')

    @property
    def hour(self):
        return self.get_dt_field('hour')

    @property
    def minute(self):
        return self.get_dt_field('minute')

    @property
    def second(self):
        return self.get_dt_field('second')

    def get_dt_field(self, field):
        out = columnops.column_empty_like_same_mask(
            self,
            dtype=np.int16
        )
        # force mask again
        out._mask = self.mask
        _gdf.apply_unaryop(self.funcs[field],
                           self,
                           out)
        return out
