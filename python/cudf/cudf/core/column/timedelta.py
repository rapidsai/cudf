# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pyarrow as pa

from cudf import _lib as libcudf
from cudf.core.column import column
from cudf.utils.dtypes import is_scalar, np_to_pa_dtype


class TimeDeltaColumn(column.ColumnBase):
    def __init__(
        self, data, dtype, mask=None, size=None, offset=0, null_count=None
    ):
        """
        Parameters
        ----------
        data : Buffer
            The datetime values
        dtype : np.dtype
            The data type
        mask : Buffer; optional
            The validity mask
        """
        dtype = np.dtype(dtype)
        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset
        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
        )
        assert self.dtype.type is np.timedelta64
        self._time_unit, _ = np.datetime_data(self.dtype)

    def to_pandas(self, index=None):
        return pd.Series(
            self.to_array(fillna="pandas").astype(self.dtype), index=index
        )

    def to_arrow(self):
        mask = None
        if self.nullable:
            mask = pa.py_buffer(self.mask_array_view.copy_to_host())
        data = pa.py_buffer(self.as_numerical.data_array_view.copy_to_host())
        pa_dtype = np_to_pa_dtype(self.dtype)
        return pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[mask, data],
            null_count=self.null_count,
        )

    @property
    def as_numerical(self):
        from cudf.core.column import build_column

        return build_column(
            data=self.base_data,
            dtype=np.int64,
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )

    def default_na_value(self):
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == "m":
            return np.timedelta64("nat", self.time_unit)
        else:
            raise TypeError(
                "datetime column of {} has no NaN value".format(self.dtype)
            )

    @property
    def time_unit(self):
        return self._time_unit

    def fillna(self, fill_value):
        if is_scalar(fill_value):
            fill_value = np.timedelta64(fill_value, self.time_unit)
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)
        print(self.dtype, fill_value.dtype)
        print(self, fill_value)
        result = libcudf.replace.replace_nulls(self, fill_value)

        return result

    def as_numerical_column(self, dtype, **kwargs):
        return self.as_numerical.astype(dtype)

    # TODO type-cast methods
