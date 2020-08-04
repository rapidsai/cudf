# Copyright (c) 2020, NVIDIA CORPORATION.
import datetime as dt
from numbers import Number

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf import _lib as libcudf
from cudf._lib.nvtx import annotate
from cudf._lib.scalar import Scalar, as_scalar
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.core.column.datetime import _numpy_to_pandas_conversion
from cudf.utils.dtypes import is_scalar, np_to_pa_dtype


class TimeDeltaColumn(column.ColumnBase):
    def __init__(
        self, data, dtype, mask=None, size=None, offset=0, null_count=None
    ):
        """
        Parameters
        ----------
        data : Buffer
            The Timedelta values
        dtype : np.dtype
            The data type
        mask : Buffer; optional
            The validity mask
        size : int, optional
            Size of memory allocation.
        offset : int
            Data offset
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.
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

        if not pd.api.types.is_timedelta64_dtype(rhs.dtype):
            if op in ("eq", "ne"):
                out_dtype = np.bool
            elif op in ("lt", "gt", "le", "ge"):
                raise TypeError(
                    f"Invalid comparison between dtype={self.dtype}"
                    f" and {type(rhs).__name__}"
                )
            elif op in ("mul", "mod"):
                out_dtype = self.dtype
            elif op == "truediv":
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
                out_dtype = determine_out_dtype(lhs.dtype, rhs.dtype)
            elif op in ("mod"):
                out_dtype = np.dtype("timedelta64[ns]")
            elif op == "truediv":
                lhs = lhs.astype("timedelta64[ns]").astype("float64")

                if isinstance(rhs, Scalar):
                    rhs = as_scalar(rhs, dtype="float64")
                else:
                    rhs = rhs.astype("timedelta64[ns]").astype("float64")
                out_dtype = np.dtype("float_")
            elif op == "floordiv":
                op = "truediv"
                lhs = lhs.astype("timedelta64[ns]").astype("float64")
                if isinstance(rhs, Scalar):
                    rhs = as_scalar(rhs, dtype="float64")
                else:
                    rhs = rhs.astype("timedelta64[ns]").astype("float64")
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
            other = other.astype("timedelta64[ns]")
            return as_scalar(other)
        elif np.isscalar(other):
            return as_scalar(other)
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

    def default_na_value(self, **kwargs):
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == "m":
            valid_scalar = kwargs.pop("valid_scalar", False)
            na_value = np.timedelta64("nat", self.time_unit)
            if valid_scalar:
                return Scalar(na_value, valid=True)
            else:
                return na_value
        else:
            raise TypeError(
                "datetime column of {} has no NaN value".format(self.dtype)
            )

    @property
    def time_unit(self):
        return self._time_unit

    def fillna(self, fill_value):
        if is_scalar(fill_value):
            if not isinstance(fill_value, Scalar):
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

    def mean(self, dtype=np.float64):
        return pd.Timedelta(
            self.as_numerical.mean(dtype=dtype), unit=self.time_unit
        )

    def median(self, dtype=np.float64):
        return pd.Timedelta(
            self.as_numerical.median(dtype=dtype), unit=self.time_unit
        )

    def quantile(self, q, interpolation, exact):
        result = self.as_numerical.quantile(
            q=q, interpolation=interpolation, exact=exact
        )
        if isinstance(q, Number):
            return [pd.Timedelta(result[0], unit=self.time_unit)]
        return result.astype(self.dtype)

    def sum(self, dtype=None):
        if len(self) == 0:
            return pd.Timedelta(None, unit=self.time_unit)
        else:
            return pd.Timedelta(
                self.as_numerical.sum(dtype=dtype), unit=self.time_unit
            )

    def components(self, index=None):
        return cudf.DataFrame(
            data={
                "days": self.binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["D"], "ns")
                    ),
                ),
                "hours": self.binary_operator(
                    "mod",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["D"], "ns")
                    ),
                ).binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["h"], "ns")
                    ),
                ),
                "minutes": self.binary_operator(
                    "mod",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["h"], "ns")
                    ),
                ).binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["m"], "ns")
                    ),
                ),
                "seconds": self.binary_operator(
                    "mod",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["m"], "ns")
                    ),
                ).binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["s"], "ns")
                    ),
                ),
                "milliseconds": self.binary_operator(
                    "mod",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["s"], "ns")
                    ),
                ).binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["ms"], "ns")
                    ),
                ),
                "microseconds": self.binary_operator(
                    "mod",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["ms"], "ns")
                    ),
                ).binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["us"], "ns")
                    ),
                ),
                "nanoseconds": self.binary_operator(
                    "mod",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["us"], "ns")
                    ),
                ).binary_operator(
                    "floordiv",
                    as_scalar(
                        np.timedelta64(_numpy_to_pandas_conversion["ns"], "ns")
                    ),
                ),
            },
            index=index,
        )

    @property
    def days(self):
        return self.binary_operator(
            "floordiv",
            as_scalar(np.timedelta64(_numpy_to_pandas_conversion["D"], "ns")),
        )

    @property
    def seconds(self):
        # This property must return the number of seconds (>= 0 and
        # less than 1 day) for each element, hence first performing
        # mod operation to remove the number of days and then performing
        # division operation to extract the number of seconds.

        return self.binary_operator(
            "mod",
            as_scalar(np.timedelta64(_numpy_to_pandas_conversion["D"], "ns")),
        ).binary_operator(
            "floordiv",
            as_scalar(np.timedelta64(_numpy_to_pandas_conversion["s"], "ns")),
        )

    @property
    def microseconds(self):
        # This property must return the number of microseconds (>= 0 and
        # less than 1 second) for each element, hence first performing
        # mod operation to remove the number of seconds and then performing
        # division operation to extract the number of microseconds.

        return self.binary_operator(
            "mod", np.timedelta64(_numpy_to_pandas_conversion["s"], "ns")
        ).binary_operator(
            "floordiv",
            as_scalar(np.timedelta64(_numpy_to_pandas_conversion["us"], "ns")),
        )

    @property
    def nanoseconds(self):
        # This property must return the number of nanoseconds (>= 0 and
        # less than 1 microsecond) for each element, hence first performing
        # mod operation to remove the number of microseconds and then
        # performing division operation to extract the number
        # of nanoseconds.

        return self.binary_operator(
            "mod",
            as_scalar(np.timedelta64(_numpy_to_pandas_conversion["us"], "ns")),
        ).binary_operator(
            "floordiv",
            as_scalar(np.timedelta64(_numpy_to_pandas_conversion["ns"], "ns")),
        )


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def binop(lhs, rhs, op, out_dtype):
    out = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)
    return out


def determine_out_dtype(lhs_dtype, rhs_dtype):
    if np.can_cast(np.dtype(lhs_dtype), np.dtype(rhs_dtype)):
        return rhs_dtype
    elif np.can_cast(np.dtype(rhs_dtype), np.dtype(lhs_dtype)):
        return lhs_dtype
    else:
        raise np.dtype("timedelta64[ns]")
