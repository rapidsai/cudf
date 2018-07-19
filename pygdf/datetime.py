import numpy as np
from libgdf_cffi import libgdf
from . import _gdf, columnops
from .buffer import Buffer
from .cudautils import compact_mask_bytes


class DatetimeColumn(columnops.TypedColumnBase):
    def __init__(self, data, mask=None, null_count=None, dtype=None):
        # currently libgdf datetime kernels fail if mask is null
        if mask is None:
            mask = np.empty(data.mem.size, dtype=np.bool)
            mask[:] = True
            mask = compact_mask_bytes(mask)
            mask = Buffer(mask)
        super(DatetimeColumn, self).__init__(data=data,
                                             mask=mask,
                                             null_count=null_count,
                                             dtype=dtype
                                             )
        # the column constructor removes mask if it's all true
        self._mask = mask


funcs = {
    'year': libgdf.gdf_extract_datetime_year,
    'month': libgdf.gdf_extract_datetime_month,
    'day': libgdf.gdf_extract_datetime_day,
    'hour': libgdf.gdf_extract_datetime_hour,
    'minute': libgdf.gdf_extract_datetime_minute,
    'second': libgdf.gdf_extract_datetime_second,
}


# def fake_extract_field(field):
#     def func(column, out):
#         data = getattr(pd.to_datetime(column.to_array()), field)
#         out._data[:] = data
#     return func


# python_funcs = {
#     'year': fake_extract_field('year')
# }


# funcs = python_funcs


class DatetimeProperties(object):
    def __init__(self, dt_column):
        # self.dt_column = weakref.ref(dt_column)
        self.dt_column = dt_column

    @property
    def year(self):
        return self.get('year')

    @property
    def month(self):
        return self.get('month')

    @property
    def day(self):
        return self.get('day')

    @property
    def hour(self):
        return self.get('hour')

    @property
    def minute(self):
        return self.get('minute')

    @property
    def second(self):
        return self.get('second')

    def get(self, field):
        out = columnops.column_empty_like_same_mask(
            self.dt_column,
            dtype=np.int16
        )
        # force mask again
        out._mask = self.dt_column.mask
        _gdf.apply_unaryop(funcs[field],
                           self.dt_column,
                           out)
        return out
