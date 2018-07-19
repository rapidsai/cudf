import numpy as np

import weakref
from . import _gdf, columnops, utils, cudautils


class DatetimeColumn(columnops.TypedColumnBase):
    pass


# funcs = {
#     'year': libgdf.gdf_extract_datetime_year
# }


def fake_extract_field(field):
    def func(column, out):
        data = getattr(pd.to_datetime(column.to_array()), field)
        out._data[:] = data
    return func


python_funcs = {
    'year': fake_extract_field('year')
}


funcs = python_funcs


class DatetimeProperties(object):
    def __init__(self, dt_column):
        # self.dt_column = weakref.ref(dt_column)
        self.dt_column = dt_column

    @property
    def year(self):
        return self.get('year')

    def get(self, field):
        out = columnops.column_empty_like_same_mask(
            self.dt_column,
            dtype=np.float64
        )
        _gdf.apply_unaryop(funcs[field],
                           self.dt_column,
                           out)
        return out
