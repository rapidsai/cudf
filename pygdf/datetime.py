import datetime as dt
import time

import numpy as np
import pandas as pd

from . import columnops, _gdf, utils
from .buffer import Buffer
from libgdf_cffi import libgdf


_unordered_impl = {
    'eq': libgdf.gdf_eq_generic,
    'ne': libgdf.gdf_ne_generic,
}


# class DatetimeColumn(numerical.NumericalColumn):
class DatetimeColumn(columnops.TypedColumnBase):
    # TODO - we only support milliseconds (date64)
    # we should support date32 and timestamp, but perhaps
    # only after we move to arrow
    # we also need to support other formats besides Date64
    funcs = {
        'year': libgdf.gdf_extract_datetime_year,
        'month': libgdf.gdf_extract_datetime_month,
        'day': libgdf.gdf_extract_datetime_day,
        'hour': libgdf.gdf_extract_datetime_hour,
        'minute': libgdf.gdf_extract_datetime_minute,
        'second': libgdf.gdf_extract_datetime_second,
    }
    _npdatetime64_dtype = np.dtype('datetime64[ms]')

    def __init__(self, data, mask=None, null_count=None, dtype=None):
        super(DatetimeColumn, self).__init__(data=data,
                                             mask=mask,
                                             null_count=null_count,
                                             dtype=dtype
                                             )
        self._precision = 1e-3
        self._inverse_precision = 1e3
        self._pandas_conversion_factor = 1e9 * self._precision

    @classmethod
    def from_numpy(cls, array):
        array = array.astype(cls._npdatetime64_dtype)
        assert array.dtype.itemsize == 8
        buf = Buffer(array)
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
        _gdf.apply_unaryop(self.funcs[field],
                           self,
                           out)
        return out

    def normalize_binop_value(self, other):

        if isinstance(other, dt.datetime):
            other = time.mktime(other.timetuple())
            ary = utils.scalar_broadcast_to(
                int(other * self._inverse_precision),
                shape=len(self),
                dtype=self._npdatetime64_dtype
            )
        elif isinstance(other, pd.Timestamp):
            ary = utils.scalar_broadcast_to(
                other.value * self._pandas_conversion_factor,
                shape=len(self),
                dtype=self._npdatetime64_dtype
            )
        elif isinstance(other, np.datetime64):
            other = other.astype(self._npdatetime64_dtype)
            ary = utils.scalar_broadcast_to(
                other,
                shape=len(self),
                dtype=self._npdatetime64_dtype
            )
        else:
            raise TypeError('cannot broadcast {}'.format(type(other)))

        buf = Buffer(ary)
        result = self.replace(data=buf, dtype=self.dtype)
        return result

    def unordered_compare(self, cmpop, rhs):
        lhs, rhs = self, rhs
        return binop(
            lhs, rhs,
            op=_unordered_impl[cmpop],
            out_dtype=np.bool
        )

    def to_pandas(self, index):
        return pd.Series(self.to_array().astype(self.dtype), index=index)


def binop(lhs, rhs, op, out_dtype):
    masked = lhs.has_null_mask or rhs.has_null_mask
    out = columnops.column_empty_like(lhs, dtype=out_dtype, masked=masked)
    null_count = _gdf.apply_binaryop(op, lhs, rhs, out)
    out = out.replace(null_count=null_count)
    return out
