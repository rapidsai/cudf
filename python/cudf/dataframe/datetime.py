import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa

from . import columnops, numerical
from cudf import _gdf
from cudf.utils import utils
from .buffer import Buffer
from libgdf_cffi import libgdf
from cudf.comm.serialize import register_distributed_serializer
from cudf._gdf import nvtx_range_push, nvtx_range_pop

_unordered_impl = {
    'eq': libgdf.gdf_eq_generic,
    'ne': libgdf.gdf_ne_generic,
}


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

    def serialize(self, serialize):
        header, frames = super(DatetimeColumn, self).serialize(serialize)
        assert 'dtype' not in header
        header['dtype'] = serialize(self._dtype)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        data, mask = cls._deserialize_data_mask(deserialize, header, frames)
        col = cls(data=data, mask=mask, null_count=header['null_count'],
                  dtype=deserialize(*header['dtype']))
        return col

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
            other = np.datetime64(other)

        if isinstance(other, pd.Timestamp):
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

    @property
    def as_numerical(self):
        return self.view(
            numerical.NumericalColumn,
            dtype='int64',
            data=self.data.astype('int64')
        )

    def astype(self, dtype):
        if self.dtype is dtype:
            return self
        return self.as_numerical.astype(dtype)

    def unordered_compare(self, cmpop, rhs):
        lhs, rhs = self, rhs
        return binop(
            lhs, rhs,
            op=_unordered_impl[cmpop],
            out_dtype=np.bool
        )

    def to_pandas(self, index):
        return pd.Series(
            self.to_array(fillna='pandas').astype(self.dtype),
            index=index
        )

    def to_arrow(self):
        mask = None
        if self.has_null_mask:
            mask = pa.py_buffer(self.nullmask.mem.copy_to_host())
        data = pa.py_buffer(self.data.mem.copy_to_host().view('int64'))
        pa_dtype = _gdf.np_to_pa_dtype(self.dtype)
        return pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[
                mask,
                data
            ],
            null_count=self.null_count
        )

    def default_na_value(self):
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == 'M':
            return np.datetime64('nat', 'ms')
        else:
            raise TypeError(
                "datetime column of {} has no NaN value".format(self.dtype))


def binop(lhs, rhs, op, out_dtype):
    nvtx_range_push("PYGDF_BINARY_OP", "orange")
    masked = lhs.has_null_mask or rhs.has_null_mask
    out = columnops.column_empty_like(lhs, dtype=out_dtype, masked=masked)
    null_count = _gdf.apply_binaryop(op, lhs, rhs, out)
    out = out.replace(null_count=null_count)
    nvtx_range_pop()
    return out


register_distributed_serializer(DatetimeColumn)
