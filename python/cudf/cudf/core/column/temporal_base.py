# Copyright (c) 2025, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf.api.types import is_scalar
from cudf.core.buffer.buffer import Buffer
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.column.numerical import NumericalColumn
from cudf.utils.dtypes import (
    CUDF_STRING_DTYPE,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    find_common_type,
)
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    import pylibcudf as plc

    from cudf._typing import ColumnLike, DtypeObj, ScalarLike
    from cudf.core.column.string import StringColumn


class TemporalBaseColumn(ColumnBase):
    """
    Base class for TimeDeltaColumn and DatetimeColumn.
    """

    _PANDAS_NA_VALUE = pd.NaT
    _UNDERLYING_DTYPE = np.dtype(np.int64)
    _NP_SCALAR: np.datetime64 | np.timedelta64
    _PD_SCALAR: pd.Timestamp | pd.Timedelta

    def __init__(
        self,
        data: Buffer,
        size: int | None,
        dtype: np.dtype | pd.DatetimeTZDtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple = (),
    ):
        if not isinstance(data, Buffer):
            raise ValueError("data must be a Buffer.")
        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset
        if len(children) != 0:
            raise ValueError(f"{type(self).__name__} must have no children.")
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def __contains__(self, item: np.datetime64 | np.timedelta64) -> bool:
        """
        Check if the column contains a given value.
        """
        return item.view(self._UNDERLYING_DTYPE) in self.astype(  # type:ignore[operator]
            self._UNDERLYING_DTYPE
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if (
            isinstance(fill_value, self._NP_SCALAR)
            and self.time_unit != np.datetime_data(fill_value)[0]
        ):
            fill_value = fill_value.astype(self.dtype)
        elif isinstance(fill_value, str) and fill_value.lower() == "nat":
            fill_value = self._NP_SCALAR(fill_value, self.time_unit)
        return super()._validate_fillna_value(fill_value)

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if isinstance(value, (np.str_, self._NP_SCALAR)):
            value = self._PD_SCALAR(value.item())
        elif isinstance(value, str):
            value = self._PD_SCALAR(value)
        elif value is pd.NaT:
            value = None
        return super()._cast_setitem_value(value)

    def _process_values_for_isin(
        self, values: Sequence
    ) -> tuple[ColumnBase, ColumnBase]:
        lhs, rhs = super()._process_values_for_isin(values)
        if len(rhs) and rhs.dtype.kind == "O":
            try:
                rhs = rhs.astype(lhs.dtype)
            except ValueError:
                pass
            else:
                warnings.warn(
                    f"The behavior of 'isin' with dtype={lhs.dtype} and "
                    "castable values (e.g. strings) is deprecated. In a "
                    "future version, these will not be considered matching "
                    "by isin. Explicitly cast to the appropriate dtype before "
                    "calling isin instead.",
                    FutureWarning,
                )
        elif isinstance(rhs, type(self)):
            rhs = rhs.astype(lhs.dtype)
        return lhs, rhs

    def _normalize_binop_operand(self, other: Any) -> pa.Scalar | ColumnBase:
        if isinstance(other, ColumnBase):
            return other
        elif self.dtype.kind == "M" and isinstance(other, cudf.DateOffset):
            return other
        elif isinstance(other, (cp.ndarray, np.ndarray)) and other.ndim == 0:
            other = other[()]

        if is_scalar(other):
            if is_na_like(other):
                return super()._normalize_binop_operand(other)
            elif self.dtype.kind == "M" and isinstance(other, pd.Timestamp):
                if other.tz is not None:
                    raise NotImplementedError(
                        "Binary operations with timezone aware operands is not supported."
                    )
                other = other.to_numpy()
            elif self.dtype.kind == "M" and isinstance(other, str):
                try:
                    other = pd.Timestamp(other)
                except ValueError:
                    return NotImplemented
            elif self.dtype.kind == "m" and isinstance(other, pd.Timedelta):
                other = other.to_numpy()
            elif isinstance(other, (np.datetime64, np.timedelta64)):
                unit = np.datetime_data(other)[0]
                if unit not in {"s", "ms", "us", "ns"}:
                    if np.isnat(other):
                        # TODO: Use self.time_unit to not modify the result resolution?
                        to_unit = "ns"
                    else:
                        to_unit = self.time_unit
                    if np.isnat(other):
                        # Workaround for https://github.com/numpy/numpy/issues/28496
                        # Once fixed, can always use the astype below
                        other = type(other)("NaT", to_unit)
                    else:
                        other = other.astype(
                            np.dtype(f"{other.dtype.kind}8[{to_unit}]")
                        )
            scalar = pa.scalar(other)
            if pa.types.is_timestamp(scalar.type):
                if scalar.type.tz is not None:
                    raise NotImplementedError(
                        "Binary operations with timezone aware operands is not supported."
                    )
                return scalar
            elif pa.types.is_duration(scalar.type):
                if self.dtype.kind == "m":
                    common_dtype = find_common_type(
                        (self.dtype, cudf_dtype_from_pa_type(scalar.type))
                    )
                    scalar = scalar.cast(cudf_dtype_to_pa_type(common_dtype))
                return scalar
            elif self.dtype.kind == "m":
                return scalar
            else:
                return NotImplemented
        return NotImplemented

    @functools.cached_property
    def time_unit(self) -> str:
        return np.datetime_data(self.dtype)[0]

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the DateTimeColumn.
        """
        raise NotImplementedError(f"cupy does not support {self.dtype}")

    def element_indexing(self, index: int) -> ScalarLike:
        result = super().element_indexing(index)
        if result is self._PANDAS_NA_VALUE:
            return result
        result = result.as_py()
        if cudf.get_option("mode.pandas_compatible"):
            return self._PD_SCALAR(result)
        elif isinstance(result, self._PD_SCALAR):
            return result.to_numpy()
        return self.dtype.type(result).astype(self.dtype, copy=False)

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        if (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(self.dtype, pd.ArrowDtype)
        ) or arrow_type:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)

        elif nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        pa_array = self.to_arrow()
        if arrow_type:
            return pd.Index(pd.arrays.ArrowExtensionArray(pa_array))
        else:
            # Workaround until the following issue is fixed:
            # https://github.com/apache/arrow/issues/45341
            return pd.Index(
                pa_array.to_numpy(zero_copy_only=False, writable=True)
            )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        col = NumericalColumn(
            data=self.base_data,  # type: ignore[arg-type]
            dtype=self._UNDERLYING_DTYPE,
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )
        return col.astype(dtype)  # type:ignore[return-value]

    def ceil(self, freq: str) -> ColumnBase:
        raise NotImplementedError("ceil is currently not implemented")

    def floor(self, freq: str) -> ColumnBase:
        raise NotImplementedError("floor is currently not implemented")

    def round(self, freq: str) -> ColumnBase:
        raise NotImplementedError("round is currently not implemented")

    def strftime(self, format: str) -> StringColumn:
        if len(self) == 0:
            return column_empty(0, dtype=CUDF_STRING_DTYPE)  # type:ignore[return-value]
        else:
            raise NotImplementedError("strftime is currently not implemented")

    def find_and_replace(
        self,
        to_replace: ColumnBase,
        replacement: ColumnBase,
        all_nan: bool = False,
    ) -> Self:
        if not isinstance(to_replace, type(self)):
            to_replace = as_column(to_replace)
            if to_replace.can_cast_safely(self.dtype):
                to_replace = to_replace.astype(self.dtype)
        if not isinstance(replacement, type(self)):
            replacement = as_column(replacement)
            if replacement.can_cast_safely(self.dtype):
                replacement = replacement.astype(self.dtype)
        if isinstance(to_replace, type(self)):
            to_replace = to_replace.astype(self._UNDERLYING_DTYPE)
        if isinstance(replacement, type(self)):
            replacement = replacement.astype(self._UNDERLYING_DTYPE)
        try:
            return (
                self.astype(self._UNDERLYING_DTYPE)  # type:ignore[return-value]
                .find_and_replace(to_replace, replacement, all_nan)
                .astype(self.dtype)
            )
        except TypeError:
            return self.copy(deep=True)

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        if to_dtype.kind == self.dtype.kind:  # type: ignore[union-attr]
            to_res, _ = np.datetime_data(to_dtype)
            max_dist = np.timedelta64(
                self.max().astype(self._UNDERLYING_DTYPE, copy=False),
                self.time_unit,
            )
            min_dist = np.timedelta64(
                self.min().astype(self._UNDERLYING_DTYPE, copy=False),
                self.time_unit,
            )
            max_to_res = np.timedelta64(
                np.iinfo(self._UNDERLYING_DTYPE).max, to_res
            ).astype(f"m8[{self.time_unit}]", copy=False)
            return bool(max_dist <= max_to_res and min_dist <= max_to_res)
        elif (
            to_dtype == self._UNDERLYING_DTYPE or to_dtype == CUDF_STRING_DTYPE
        ):
            # can safely cast to representation, or string
            return True
        else:
            return False

    def mean(
        self, skipna: bool | None = None, min_count: int = 0
    ) -> pd.Timestamp | pd.Timedelta:
        return self._PD_SCALAR(
            self.astype(self._UNDERLYING_DTYPE).mean(  # type:ignore[call-arg]
                skipna=skipna, min_count=min_count
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def std(
        self, skipna: bool | None = None, min_count: int = 0, ddof: int = 1
    ) -> pd.Timedelta:
        return pd.Timedelta(
            self.astype(self._UNDERLYING_DTYPE).std(  # type:ignore[call-arg]
                skipna=skipna, min_count=min_count, ddof=ddof
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def median(
        self, skipna: bool | None = None
    ) -> pd.Timestamp | pd.Timedelta:
        return self._PD_SCALAR(
            self.astype(self._UNDERLYING_DTYPE).median(skipna=skipna),  # type:ignore[call-arg]
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def cov(self, other: Self) -> float:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"cannot perform cov with types {self.dtype}, {other.dtype}"
            )
        return self.astype(self._UNDERLYING_DTYPE).cov(  # type:ignore[attr-defined]
            other.astype(self._UNDERLYING_DTYPE)
        )

    def corr(self, other: Self) -> float:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"cannot perform corr with types {self.dtype}, {other.dtype}"
            )
        return self.astype(self._UNDERLYING_DTYPE).corr(  # type:ignore[attr-defined]
            other.astype(self._UNDERLYING_DTYPE)
        )

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase | pd.Timestamp | pd.Timedelta:
        result = self.astype(self._UNDERLYING_DTYPE).quantile(
            q=q,
            interpolation=interpolation,
            exact=exact,
            return_scalar=return_scalar,
        )
        if return_scalar:
            return self._PD_SCALAR(result, unit=self.time_unit).as_unit(
                self.time_unit
            )
        return result.astype(self.dtype)
