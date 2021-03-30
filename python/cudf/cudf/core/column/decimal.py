# Copyright (c) 2021, NVIDIA CORPORATION.

from typing import cast

from pandas.api.types import is_integer_dtype

import cudf
from cudf import _lib as libcudf
from cudf._lib.strings.convert.convert_fixed_point import (
    from_decimal as cpp_from_decimal,
)
from cudf._typing import Dtype
from cudf.core.column import ColumnBase, as_column
from cudf.core.dtypes import Decimal64Dtype


class DecimalColumn(ColumnBase):
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
