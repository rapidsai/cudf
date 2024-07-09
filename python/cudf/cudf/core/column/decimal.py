# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from __future__ import annotations

import warnings
from decimal import Decimal
from typing import TYPE_CHECKING, Sequence, cast

import cupy as cp
import numpy as np
import pyarrow as pa

import cudf
from cudf import _lib as libcudf
from cudf._lib.strings.convert.convert_fixed_point import (
    from_decimal as cpp_from_decimal,
)
from cudf.api.types import is_integer_dtype, is_scalar
from cudf.core.buffer import as_buffer
from cudf.core.column import ColumnBase
from cudf.core.dtypes import (
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    DecimalDtype,
)
from cudf.core.mixins import BinaryOperand
from cudf.utils.utils import pa_mask_buffer_to_mask

from .numerical_base import NumericalBaseColumn

if TYPE_CHECKING:
    from cudf._typing import ColumnBinaryOperand, ColumnLike, Dtype, ScalarLike


class DecimalBaseColumn(NumericalBaseColumn):
    """Base column for decimal32, decimal64 or decimal128 columns"""

    dtype: DecimalDtype
    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Decimals are not yet supported via `__cuda_array_interface__`"
        )

    def as_decimal_column(
        self,
        dtype: Dtype,
    ) -> "DecimalBaseColumn":
        if (
            isinstance(dtype, cudf.core.dtypes.DecimalDtype)
            and dtype.scale < self.dtype.scale
        ):
            warnings.warn(
                "cuDF truncates when downcasting decimals to a lower scale. "
                "To round, use Series.round() or DataFrame.round()."
            )

        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype)

    def as_string_column(
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.StringColumn":
        if len(self) > 0:
            return cpp_from_decimal(self)
        else:
            return cast(
                cudf.core.column.StringColumn,
                cudf.core.column.column_empty(0, dtype="object"),
            )

    def __pow__(self, other):
        if isinstance(other, int):
            if other == 0:
                res = cudf.core.column.as_column(
                    1, dtype=self.dtype, length=len(self)
                )
                if self.nullable:
                    res = res.set_mask(self.mask)
                return res
            elif other < 0:
                raise TypeError("Power of negative integers not supported.")
            res = self
            for _ in range(other - 1):
                res = self * res
            return res
        else:
            raise NotImplementedError(
                f"__pow__ of types {self.dtype} and {type(other)} is "
                "not yet implemented."
            )

    # Decimals in libcudf don't support truediv, see
    # https://github.com/rapidsai/cudf/pull/7435 for explanation.
    def __truediv__(self, other):
        return self._binaryop(other, "__div__")

    def __rtruediv__(self, other):
        return self._binaryop(other, "__rdiv__")

    def _binaryop(self, other: ColumnBinaryOperand, op: str):
        reflect, op = self._check_reflected_op(op)
        other = self._wrap_binop_normalization(other)
        if other is NotImplemented:
            return NotImplemented
        lhs, rhs = (other, self) if reflect else (self, other)

        # Binary Arithmetics between decimal columns. `Scale` and `precision`
        # are computed outside of libcudf
        if op in {"__add__", "__sub__", "__mul__", "__div__"}:
            output_type = _get_decimal_type(lhs.dtype, rhs.dtype, op)
            result = libcudf.binaryop.binaryop(lhs, rhs, op, output_type)
            # TODO:  Why is this necessary? Why isn't the result's
            # precision already set correctly based on output_type?
            result.dtype.precision = output_type.precision
        elif op in {
            "__eq__",
            "__ne__",
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
        }:
            result = libcudf.binaryop.binaryop(lhs, rhs, op, bool)
        else:
            raise TypeError(
                f"{op} not supported for the following dtypes: "
                f"{self.dtype}, {other.dtype}"
            )

        return result

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> cudf.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if isinstance(fill_value, (int, Decimal)):
            return cudf.Scalar(fill_value, dtype=self.dtype)
        elif isinstance(fill_value, ColumnBase) and (
            isinstance(self.dtype, DecimalDtype) or self.dtype.kind in "iu"
        ):
            return fill_value.astype(self.dtype)
        raise TypeError(
            "Decimal columns only support using fillna with decimal and "
            "integer values"
        )

    def normalize_binop_value(self, other):
        if isinstance(other, ColumnBase):
            if isinstance(other, cudf.core.column.NumericalColumn):
                if not is_integer_dtype(other.dtype):
                    raise TypeError(
                        "Decimal columns only support binary operations with "
                        "integer numerical columns."
                    )
                other = other.astype(
                    self.dtype.__class__(self.dtype.__class__.MAX_PRECISION, 0)
                )
            elif not isinstance(other, DecimalBaseColumn):
                return NotImplemented
            elif not isinstance(self.dtype, other.dtype.__class__):
                # This branch occurs if we have a DecimalBaseColumn of a
                # different size (e.g. 64 instead of 32).
                if _same_precision_and_scale(self.dtype, other.dtype):
                    other = other.astype(self.dtype)
            return other
        if isinstance(other, cudf.Scalar) and isinstance(
            # TODO: Should it be possible to cast scalars of other numerical
            # types to decimal?
            other.dtype,
            cudf.core.dtypes.DecimalDtype,
        ):
            if _same_precision_and_scale(self.dtype, other.dtype):
                other = other.astype(self.dtype)
            return other
        elif is_scalar(other) and isinstance(other, (int, Decimal)):
            other = Decimal(other)
            metadata = other.as_tuple()
            precision = max(len(metadata.digits), metadata.exponent)
            scale = -metadata.exponent
            return cudf.Scalar(
                other, dtype=self.dtype.__class__(precision, scale)
            )
        return NotImplemented

    def _decimal_quantile(
        self, q: float | Sequence[float], interpolation: str, exact: bool
    ) -> ColumnBase:
        quant = [float(q)] if not isinstance(q, (Sequence, np.ndarray)) else q
        # get sorted indices and exclude nulls
        indices = libcudf.sort.order_by(
            [self], [True], "first", stable=True
        ).slice(self.null_count, len(self))
        result = libcudf.quantiles.quantile(
            self, quant, interpolation, indices, exact
        )
        return result._with_type_metadata(self.dtype)

    def as_numerical_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.NumericalColumn":
        return libcudf.unary.cast(self, dtype)


class Decimal32Column(DecimalBaseColumn):
    dtype: Decimal32Dtype

    @classmethod
    def from_arrow(cls, data: pa.Array):
        dtype = Decimal32Dtype.from_arrow(data.type)
        mask_buf = data.buffers()[0]
        mask = (
            mask_buf
            if mask_buf is None
            else pa_mask_buffer_to_mask(mask_buf, len(data))
        )
        data_128 = cp.array(np.frombuffer(data.buffers()[1]).view("int32"))
        data_32 = data_128[::4].copy()
        return cls(
            data=as_buffer(data_32.view("uint8")),
            size=len(data),
            dtype=dtype,
            offset=data.offset,
            mask=mask,
        )

    def to_arrow(self):
        data_buf_32 = np.array(self.base_data.memoryview()).view("int32")
        data_buf_128 = np.empty(len(data_buf_32) * 4, dtype="int32")

        # use striding to set the first 32 bits of each 128-bit chunk:
        data_buf_128[::4] = data_buf_32
        # use striding again to set the remaining bits of each 128-bit chunk:
        # 0 for non-negative values, -1 for negative values:
        data_buf_128[1::4] = np.piecewise(
            data_buf_32, [data_buf_32 < 0], [-1, 0]
        )
        data_buf_128[2::4] = np.piecewise(
            data_buf_32, [data_buf_32 < 0], [-1, 0]
        )
        data_buf_128[3::4] = np.piecewise(
            data_buf_32, [data_buf_32 < 0], [-1, 0]
        )
        data_buf = pa.py_buffer(data_buf_128)
        mask_buf = (
            self.base_mask
            if self.base_mask is None
            else pa.py_buffer(self.base_mask.memoryview())
        )
        return pa.Array.from_buffers(
            type=self.dtype.to_arrow(),
            offset=self._offset,
            length=self.size,
            buffers=[mask_buf, data_buf],
        )

    def _with_type_metadata(
        self: "cudf.core.column.Decimal32Column", dtype: Dtype
    ) -> "cudf.core.column.Decimal32Column":
        if isinstance(dtype, Decimal32Dtype):
            self.dtype.precision = dtype.precision

        return self


class Decimal128Column(DecimalBaseColumn):
    dtype: Decimal128Dtype

    @classmethod
    def from_arrow(cls, data: pa.Array):
        result = cast(Decimal128Dtype, super().from_arrow(data))
        result.dtype.precision = data.type.precision
        return result

    def to_arrow(self):
        return super().to_arrow().cast(self.dtype.to_arrow())

    def _with_type_metadata(
        self: "cudf.core.column.Decimal128Column", dtype: Dtype
    ) -> "cudf.core.column.Decimal128Column":
        if isinstance(dtype, Decimal128Dtype):
            self.dtype.precision = dtype.precision

        return self


class Decimal64Column(DecimalBaseColumn):
    dtype: Decimal64Dtype

    def __setitem__(self, key, value):
        if isinstance(value, np.integer):
            value = int(value)
        super().__setitem__(key, value)

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
            data=as_buffer(data_64.view("uint8")),
            size=len(data),
            dtype=dtype,
            offset=data.offset,
            mask=mask,
        )

    def to_arrow(self):
        data_buf_64 = np.array(self.base_data.memoryview()).view("int64")
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
            else pa.py_buffer(self.base_mask.memoryview())
        )
        return pa.Array.from_buffers(
            type=self.dtype.to_arrow(),
            offset=self._offset,
            length=self.size,
            buffers=[mask_buf, data_buf],
        )

    def _with_type_metadata(
        self: "cudf.core.column.Decimal64Column", dtype: Dtype
    ) -> "cudf.core.column.Decimal64Column":
        if isinstance(dtype, Decimal64Dtype):
            self.dtype.precision = dtype.precision

        return self


def _get_decimal_type(lhs_dtype, rhs_dtype, op):
    """
    Returns the resulting decimal type after calculating
    precision & scale when performing the binary operation
    `op` for the given dtypes.

    For precision & scale calculations see : https://docs.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql
    """  # noqa: E501

    # This should at some point be hooked up to libcudf's
    # binary_operation_fixed_point_scale

    p1, p2 = lhs_dtype.precision, rhs_dtype.precision
    s1, s2 = lhs_dtype.scale, rhs_dtype.scale

    if op in {"__add__", "__sub__"}:
        scale = max(s1, s2)
        precision = scale + max(p1 - s1, p2 - s2) + 1
        if precision > Decimal128Dtype.MAX_PRECISION:
            precision = Decimal128Dtype.MAX_PRECISION
            scale = Decimal128Dtype.MAX_PRECISION - max(p1 - s1, p2 - s2)
    elif op in {"__mul__", "__div__"}:
        if op == "__mul__":
            scale = s1 + s2
            precision = p1 + p2 + 1
        else:
            scale = max(6, s1 + p2 + 1)
            precision = p1 - s1 + s2 + scale
        if precision > Decimal128Dtype.MAX_PRECISION:
            integral = precision - scale
            if integral < 32:
                scale = min(scale, Decimal128Dtype.MAX_PRECISION - integral)
            elif scale > 6 and integral > 32:
                scale = 6
            precision = Decimal128Dtype.MAX_PRECISION
    else:
        raise NotImplementedError()

    try:
        if isinstance(lhs_dtype, type(rhs_dtype)):
            # SCENARIO 1: If `lhs_dtype` & `rhs_dtype` are same, then try to
            # see if `precision` & `scale` can be fit into this type.
            return lhs_dtype.__class__(precision=precision, scale=scale)
        else:
            # SCENARIO 2: If `lhs_dtype` & `rhs_dtype` are of different dtypes,
            # then try to see if `precision` & `scale` can be fit into the type
            # with greater MAX_PRECISION (i.e., the bigger dtype).
            if lhs_dtype.MAX_PRECISION >= rhs_dtype.MAX_PRECISION:
                return lhs_dtype.__class__(precision=precision, scale=scale)
            else:
                return rhs_dtype.__class__(precision=precision, scale=scale)
    except ValueError:
        # Call to _validate fails, which means we need
        # to goto SCENARIO 3.
        pass

    # SCENARIO 3: If either of the above two scenarios fail, then get the
    # MAX_PRECISION of `lhs_dtype` & `rhs_dtype` so that we can only check
    # and return a dtype that is greater than or equal to input dtype that
    # can fit `precision` & `scale`.
    max_precision = max(lhs_dtype.MAX_PRECISION, rhs_dtype.MAX_PRECISION)
    for decimal_type in (
        Decimal32Dtype,
        Decimal64Dtype,
        Decimal128Dtype,
    ):
        if decimal_type.MAX_PRECISION >= max_precision:
            try:
                return decimal_type(precision=precision, scale=scale)
            except ValueError:
                # Call to _validate fails, which means we need
                # to try the next dtype
                continue

    # if we've reached this point, we cannot create a decimal type without
    # overflow; raise an informative error
    raise ValueError(
        f"Performing {op} between columns of type {repr(lhs_dtype)} and "
        f"{repr(rhs_dtype)} would result in overflow"
    )


def _same_precision_and_scale(lhs: DecimalDtype, rhs: DecimalDtype) -> bool:
    return lhs.precision == rhs.precision and lhs.scale == rhs.scale
