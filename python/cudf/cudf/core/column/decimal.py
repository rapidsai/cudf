# Copyright (c) 2021, NVIDIA CORPORATION.

import cudf
import cupy as cp
import numpy as np
import pyarrow as pa
from pandas.api.types import is_integer_dtype
from typing import cast

from cudf import _lib as libcudf
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase
from cudf.core.dtypes import Decimal64Dtype
from cudf.utils.utils import pa_mask_buffer_to_mask

from cudf._typing import Dtype
from cudf._lib.strings.convert.convert_fixed_point import (
    from_decimal as cpp_from_decimal,
)
from cudf.core.column import as_column


class DecimalColumn(ColumnBase):
    @classmethod
    def from_arrow(cls, data: pa.Array):
        dtype = Decimal64Dtype.from_arrow(data.type)
        mask_buf = data.buffers()[0]
        mask = (
            mask_buf
            if mask_buf is None
            else pa_mask_buffer_to_mask(mask_buf, len(data))
        )
        data_128 = cp.array(np.frombuffer(data.buffers()[1]).view("int64"))
        data_64 = data_128[::2].copy()
        return cls(
            data=Buffer(data_64.view("uint8")),
            size=len(data),
            dtype=dtype,
            mask=mask,
        )

    def to_arrow(self):
        data_buf_64 = self.base_data.to_host_array().view("int64")
        data_buf_128 = np.empty(len(data_buf_64) * 2, dtype="int64")
        # use striding to set the first 64 bits of each 128-bit chunk:
        data_buf_128[::2] = data_buf_64
        # use striding again to set the remaining bits of each 128-bit chunk:
        # 0 for non-negative values, -1 for negative values:
        data_buf_128[1::2] = np.piecewise(
            data_buf_64, [data_buf_64 < 0], [-1, 0]
        )
        data_buf = pa.py_buffer(data_buf_128)
        mask_buf = (
            self.base_mask
            if self.base_mask is None
            else pa.py_buffer(self.base_mask.to_host_array())
        )
        return pa.Array.from_buffers(
            type=self.dtype.to_arrow(),
            length=self.size,
            buffers=[mask_buf, data_buf],
        )

    def binary_operator(self, op, other, reflect=False):
        if reflect:
            self, other = other, self
        scale = _binop_scale(self.dtype, other.dtype, op)
        output_type = Decimal64Dtype(
            scale=scale, precision=Decimal64Dtype.MAX_PRECISION
        )  # precision will be ignored, libcudf has no notion of precision
        result = libcudf.binaryop.binaryop(self, other, op, output_type)
        result.dtype.precision = _binop_precision(self.dtype, other.dtype, op)
        return result

    def _apply_scan_op(self, op: str) -> ColumnBase:
        return libcudf.reduce.scan(op, self, True)

    def as_decimal_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.DecimalColumn":
        if dtype == self.dtype:
            return self
        result = libcudf.unary.cast(self, dtype)
        if isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
            result.dtype.precision = dtype.precision
        return result

    def as_numerical_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.NumericalColumn":
        if is_integer_dtype(dtype):
            raise NotImplementedError(
                "Casting from decimal types to integer "
                "types not currently supported"
            )
        return libcudf.unary.cast(self, dtype)

    def as_string_column(
        self, dtype: Dtype, format=None
    ) -> "cudf.core.column.StringColumn":
        if len(self) > 0:
            return cpp_from_decimal(self)
        else:
            return cast(
                "cudf.core.column.StringColumn", as_column([], dtype="object")
            )


def _binop_scale(l_dtype, r_dtype, op):
    # This should at some point be hooked up to libcudf's
    # binary_operation_fixed_point_scale
    s1, s2 = l_dtype.scale, r_dtype.scale
    if op in ("add", "sub"):
        return max(s1, s2)
    elif op == "mul":
        return s1 + s2
    else:
        raise NotImplementedError()


def _binop_precision(l_dtype, r_dtype, op):
    """
    Returns the result precision when performing the
    binary operation `op` for the given dtypes.

    See: https://docs.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql
    """  # noqa: E501
    p1, p2 = l_dtype.precision, r_dtype.precision
    s1, s2 = l_dtype.scale, r_dtype.scale
    if op in ("add", "sub"):
        return max(s1, s2) + max(p1 - s1, p2 - s2) + 1
    elif op == "mul":
        return p1 + p2 + 1
    else:
        raise NotImplementedError()
