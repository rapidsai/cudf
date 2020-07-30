# Copyright (c) 2020, NVIDIA CORPORATION.
import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa

from cudf import _lib as libcudf
from cudf._lib.nvtx import annotate
from cudf._lib.scalar import Scalar, as_scalar
from cudf.core.buffer import Buffer
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

    def __contains__(self, item):
        # Handles improper item types
        try:
            item = np.timedelta64(item, self._time_unit)
        except Exception:
            return False
        return item.astype("int_") in self.as_numerical

    def to_pandas(self, index=None, nullable_pd_dtype=False):
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

    def binary_operator(self, op, rhs, reflect=False):
        lhs, rhs = self, rhs

        if isinstance(rhs, Scalar) or np.isscalar(rhs):
            if op in ("eq", "ne"):
                out_dtype = np.bool
            elif op in ("lt", "gt", "le", "ge"):
                raise TypeError(
                    f"Invalid comparison between dtype={self.dtype}"
                    f" and {type(rhs).__name__}"
                )
            elif op in ("mul", "mod", "truediv"):
                out_dtype = self.dtype
            elif op == "floordiv":
                op = "truediv"
                out_dtype = self.dtype
            elif op in ("add", "sub"):
                out_dtype = self.dtype
                if isinstance(rhs, Scalar) and rhs.value is None:
                    return column.column_empty(
                        row_count=len(self), dtype=self.dtype, masked=True
                    )
            else:
                raise TypeError(
                    f"Series of dtype {self.dtype} cannot perform "
                    f" the operation {op}"
                )
        else:
            if op in ("eq", "ne", "lt", "gt", "le", "ge"):
                out_dtype = np.bool
            elif op in ("add", "sub"):
                out_dtype = self.dtype
            elif op == "truediv":
                lhs = lhs.astype("float64")
                out_dtype = np.dtype("float_")
            elif op == "floordiv":
                op = "truediv"
                out_dtype = np.dtype("int_")
            else:
                raise TypeError(
                    f"Series of dtype {self.dtype} cannot perform "
                    f" the operation {op}"
                )

        if reflect:
            lhs, rhs = rhs, lhs
        return binop(lhs, rhs, op=op, out_dtype=out_dtype)

    def normalize_binop_value(self, other):
        if isinstance(other, dt.timedelta):
            other = np.timedelta64(other)

        if isinstance(other, pd.Timestamp):
            # TODO
            pass
        elif isinstance(other, np.timedelta64):
            other = other.astype(self.dtype)
            return as_scalar(other)
        elif np.isscalar(other):
            return other
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))
        ary = None
        return column.build_column(
            data=Buffer(ary.data_array_view.view("|u1")), dtype=self.dtype
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

        result = libcudf.replace.replace_nulls(self, fill_value)

        return result

    def as_numerical_column(self, dtype, **kwargs):
        return self.as_numerical.astype(dtype)

    def as_datetime_column(self, dtype, **kwargs):
        raise TypeError(
            f"cannot astype a timedelta from [{self.dtype}] to [{dtype}]"
        )

    def as_string_column(self, dtype, **kwargs):
        # TODO: To be implemented once
        # https://github.com/rapidsai/cudf/pull/5625/
        # is merged.
        raise NotImplementedError

    def as_timedelta_column(self, dtype, **kwargs):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype=dtype)


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def binop(lhs, rhs, op, out_dtype):
    out = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)
    return out
