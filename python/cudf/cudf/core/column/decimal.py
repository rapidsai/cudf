# Copyright (c) 2021, NVIDIA CORPORATION.

from decimal import Decimal
from typing import Any, Sequence, Tuple, Union, cast
from warnings import warn

import cupy as cp
import numpy as np
import pyarrow as pa

import cudf
from cudf import _lib as libcudf
from cudf._lib.quantiles import quantile as cpp_quantile
from cudf._lib.strings.convert.convert_fixed_point import (
    from_decimal as cpp_from_decimal,
)
from cudf._typing import Dtype
from cudf.api.types import is_integer_dtype
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase, as_column
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype
from cudf.utils.dtypes import is_scalar
from cudf.utils.utils import pa_mask_buffer_to_mask

from .numerical_base import NumericalBaseColumn


class Decimal32Column(NumericalBaseColumn):
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
            data=Buffer(data_32.view("uint8")),
            size=len(data),
            dtype=dtype,
            offset=data.offset,
            mask=mask,
        )

    def to_arrow(self):
        data_buf_32 = self.base_data.to_host_array().view("int32")
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
            else pa.py_buffer(self.base_mask.to_host_array())
        )
        return pa.Array.from_buffers(
            type=self.dtype.to_arrow(),
            offset=self._offset,
            length=self.size,
            buffers=[mask_buf, data_buf],
        )


class Decimal64Column(NumericalBaseColumn):
    dtype: Decimal64Dtype

    def __truediv__(self, other):
        # TODO: This override is not sufficient. While it will change the
        # behavior of x / y for two decimal columns, it will not affect
        # col1.binary_operator(col2), which is how Series/Index will call this.
        return self.binary_operator("div", other)

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
            data=Buffer(data_64.view("uint8")),
            size=len(data),
            dtype=dtype,
            offset=data.offset,
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
            offset=self._offset,
            length=self.size,
            buffers=[mask_buf, data_buf],
        )

    def binary_operator(self, op, other, reflect=False):
        if reflect:
            self, other = other, self

        # Binary Arithmatics between decimal columns. `Scale` and `precision`
        # are computed outside of libcudf
        if op in ("add", "sub", "mul", "div"):
            scale = _binop_scale(self.dtype, other.dtype, op)
            output_type = Decimal64Dtype(
                scale=scale, precision=Decimal64Dtype.MAX_PRECISION
            )  # precision will be ignored, libcudf has no notion of precision
            result = libcudf.binaryop.binaryop(self, other, op, output_type)
            result.dtype.precision = _binop_precision(
                self.dtype, other.dtype, op
            )
        elif op in ("eq", "ne", "lt", "gt", "le", "ge"):
            if not isinstance(
                other,
                (
                    Decimal64Column,
                    cudf.core.column.NumericalColumn,
                    cudf.Scalar,
                ),
            ):
                raise TypeError(
                    f"Operator {op} not supported between"
                    f"{str(type(self))} and {str(type(other))}"
                )
            if isinstance(
                other, cudf.core.column.NumericalColumn
            ) and not is_integer_dtype(other.dtype):
                raise TypeError(
                    f"Only decimal and integer column is supported for {op}."
                )
            if isinstance(other, cudf.core.column.NumericalColumn):
                other = other.as_decimal_column(
                    Decimal64Dtype(Decimal64Dtype.MAX_PRECISION, 0)
                )
            result = libcudf.binaryop.binaryop(self, other, op, bool)
        return result

    def normalize_binop_value(self, other):
        if is_scalar(other) and isinstance(other, (int, np.int, Decimal)):
            return cudf.Scalar(Decimal(other))
        elif isinstance(other, cudf.Scalar) and isinstance(
            other.dtype, cudf.Decimal64Dtype
        ):
            return other
        else:
            raise TypeError(f"cannot normalize {type(other)}")

    def _decimal_quantile(
        self, q: Union[float, Sequence[float]], interpolation: str, exact: bool
    ) -> ColumnBase:
        quant = [float(q)] if not isinstance(q, (Sequence, np.ndarray)) else q
        # get sorted indices and exclude nulls
        sorted_indices = self.as_frame()._get_sorted_inds(
            ascending=True, na_position="first"
        )
        sorted_indices = sorted_indices[self.null_count :]

        result = cpp_quantile(
            self, quant, interpolation, sorted_indices, exact
        )

        return result._with_type_metadata(self.dtype)

    def as_decimal_column(
        self, dtype: Dtype, **kwargs
    ) -> Union[
        "cudf.core.column.Decimal32Column", "cudf.core.column.Decimal64Column"
    ]:
        if (
            isinstance(dtype, Decimal64Dtype)
            and dtype.scale < self.dtype.scale
        ):
            warn(
                "cuDF truncates when downcasting decimals to a lower scale. "
                "To round, use Series.round() or DataFrame.round()."
            )

        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype)

    def as_numerical_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.NumericalColumn":
        return libcudf.unary.cast(self, dtype)

    def as_string_column(
        self, dtype: Dtype, format=None, **kwargs
    ) -> "cudf.core.column.StringColumn":
        if len(self) > 0:
            return cpp_from_decimal(self)
        else:
            return cast(
                "cudf.core.column.StringColumn", as_column([], dtype="object")
            )

    def fillna(
        self, value: Any = None, method: str = None, dtype: Dtype = None
    ):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if isinstance(value, (int, Decimal)):
            value = cudf.Scalar(value, dtype=self.dtype)
        elif (
            isinstance(value, Decimal64Column)
            or isinstance(value, cudf.core.column.NumericalColumn)
            and is_integer_dtype(value.dtype)
        ):
            value = value.astype(self.dtype)
        else:
            raise TypeError(
                "Decimal columns only support using fillna with decimal and "
                "integer values"
            )

        result = libcudf.replace.replace_nulls(
            input_col=self, replacement=value, method=method, dtype=dtype
        )
        return result._with_type_metadata(self.dtype)

    def serialize(self) -> Tuple[dict, list]:
        header, frames = super().serialize()
        header["dtype"] = self.dtype.serialize()
        header["size"] = self.size
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> ColumnBase:
        dtype = cudf.Decimal64Dtype.deserialize(*header["dtype"])
        header["dtype"] = dtype
        return super().deserialize(header, frames)

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Decimals are not yet supported via `__cuda_array_interface__`"
        )

    def _with_type_metadata(
        self: "cudf.core.column.Decimal64Column", dtype: Dtype
    ) -> "cudf.core.column.Decimal64Column":
        if isinstance(dtype, Decimal64Dtype):
            self.dtype.precision = dtype.precision

        return self


def _binop_scale(l_dtype, r_dtype, op):
    # This should at some point be hooked up to libcudf's
    # binary_operation_fixed_point_scale
    s1, s2 = l_dtype.scale, r_dtype.scale
    if op in ("add", "sub"):
        return max(s1, s2)
    elif op == "mul":
        return s1 + s2
    elif op == "div":
        return s1 - s2
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
        result = max(s1, s2) + max(p1 - s1, p2 - s2) + 1
    elif op in ("mul", "div"):
        result = p1 + p2 + 1
    else:
        raise NotImplementedError()

    return min(result, cudf.Decimal64Dtype.MAX_PRECISION)
