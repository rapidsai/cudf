import weakref
from . import _gdf, columnops, utils, cudautils


class DatetimeColumn(columnops.TypedColumnBase):
    pass


class DatetimeProperties(object):
    def __init__(self, dt_column):
        self.dt_column = weakref.ref(dt_column)
