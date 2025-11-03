# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Define an interface for columns that can perform numerical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

import pylibcudf as plc

import cudf
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column.column import ColumnBase, column_empty
from cudf.core.missing import NA
from cudf.core.mixins import Scannable
from cudf.utils.dtypes import _get_nan_for_dtype

if TYPE_CHECKING:
    from cudf._typing import ScalarLike


_unaryop_map = {
    "ASIN": "ARCSIN",
    "ACOS": "ARCCOS",
    "ATAN": "ARCTAN",
    "INVERT": "BIT_INVERT",
}


class NumericalBaseColumn(ColumnBase, Scannable):
    """
    A column composed of numerical (bool, integer, float, decimal) data.

    This class encodes a standard interface for different types of columns
    containing numerical types of data. In particular, mathematical operations
    that make sense whether a column is integral or real, fixed or floating
    point, should be encoded here.
    """

    _VALID_REDUCTIONS = {
        "sum",
        "product",
        "sum_of_squares",
        "mean",
        "var",
        "std",
    }

    _VALID_SCANS = {
        "cumsum",
        "cumprod",
        "cummin",
        "cummax",
    }

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls()

    def kurtosis(self, skipna: bool = True) -> float:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if len(self) == 0 or self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]

        self = self.nans_to_nulls().dropna()

        if len(self) < 4:
            return _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]

        n = len(self)
        miu = self.mean()
        m4_numerator = ((self - miu) ** 4).sum()
        V = self.var()

        if V == 0:
            return np.float64(0)

        term_one_section_one = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        term_one_section_two = m4_numerator / (V**2)
        term_two = ((n - 1) ** 2) / ((n - 2) * (n - 3))
        kurt = term_one_section_one * term_one_section_two - 3 * term_two
        return kurt

    def skew(self, skipna: bool = True) -> ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if len(self) == 0 or self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        self = self.nans_to_nulls().dropna()

        if len(self) < 3:
            return _get_nan_for_dtype(self.dtype)

        n = len(self)
        miu = self.mean()
        m3 = (((self - miu) ** 3).sum()) / n
        m2 = self.var(ddof=0)

        if m2 == 0:
            return np.float64(0)

        unbiased_coef = ((n * (n - 1)) ** 0.5) / (n - 2)
        skew = unbiased_coef * m3 / (m2 ** (3 / 2))
        return skew

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> NumericalBaseColumn:
        if np.logical_or(q < 0, q > 1).any():
            raise ValueError(
                "percentiles should all be in the interval [0, 1]"
            )
        # Beyond this point, q either being scalar or list-like
        # will only have values in range [0, 1]
        if len(self) == 0:
            result = cast(
                NumericalBaseColumn,
                column_empty(row_count=len(q), dtype=self.dtype),
            )
        else:
            no_nans = self.nans_to_nulls()
            # get sorted indices and exclude nulls
            indices = (
                no_nans.argsort(ascending=True, na_position="first")
                .slice(no_nans.null_count, len(no_nans))
                .astype(np.dtype(np.int32))
            )
            with acquire_spill_lock():
                plc_column = plc.quantiles.quantile(
                    no_nans.to_pylibcudf(mode="read"),
                    q,
                    plc.types.Interpolation[interpolation.upper()],
                    indices.to_pylibcudf(mode="read"),
                    exact,
                )
                result = type(self).from_pylibcudf(plc_column)
        if return_scalar:
            scalar_result = result.element_indexing(0)
            if interpolation in {"lower", "higher", "nearest"}:
                try:
                    new_scalar = self.dtype.type(scalar_result)
                    scalar_result = (
                        new_scalar
                        if new_scalar == scalar_result
                        else scalar_result
                    )
                except (TypeError, ValueError):
                    pass
            return (
                _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]
                if scalar_result is NA
                else scalar_result
            )
        return result

    def mean(
        self,
        skipna: bool = True,
        min_count: int = 0,
    ) -> ScalarLike:
        return self._reduce("mean", skipna=skipna, min_count=min_count)

    def var(
        self,
        skipna: bool = True,
        min_count: int = 0,
        ddof: int = 1,
    ) -> ScalarLike:
        result = self._reduce(
            "var", skipna=skipna, min_count=min_count, ddof=ddof
        )
        if result is NA:
            return _get_nan_for_dtype(self.dtype)
        return result

    def std(
        self,
        skipna: bool = True,
        min_count: int = 0,
        ddof: int = 1,
    ) -> ScalarLike:
        result = self._reduce(
            "std", skipna=skipna, min_count=min_count, ddof=ddof
        )
        if result is NA:
            return _get_nan_for_dtype(self.dtype)
        return result

    def median(self, skipna: bool = True) -> NumericalBaseColumn:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]

        # enforce linear in case the default ever changes
        result = self.quantile(
            np.array([0.5]),
            interpolation="linear",
            exact=True,
            return_scalar=True,
        )
        if self.dtype.kind == "f":
            result = self.dtype.type(result)
        return result

    def cov(self, other: NumericalBaseColumn) -> float:
        if (
            len(self) == 0
            or len(other) == 0
            or (len(self) == 1 and len(other) == 1)
        ):
            return _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]

        result = (self - self.mean()) * (other - other.mean())
        cov_sample = result.sum() / (len(self) - 1)
        return cov_sample

    def corr(self, other: NumericalBaseColumn) -> float:
        if len(self) == 0 or len(other) == 0:
            return _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]

        cov = self.cov(other)
        lhs_std, rhs_std = self.std(), other.std()

        if not cov or lhs_std == 0 or rhs_std == 0:
            return _get_nan_for_dtype(self.dtype)  # type: ignore[return-value]
        return cov / lhs_std / rhs_std

    def round(
        self,
        decimals: int = 0,
        how: Literal["half_even", "half_up"] = "half_even",
    ) -> NumericalBaseColumn:
        if not cudf.api.types.is_integer(decimals):
            raise TypeError("Argument 'decimals' must an integer")
        if how not in {"half_even", "half_up"}:
            raise ValueError(f"{how=} must be either 'half_even' or 'half_up'")
        plc_how = plc.round.RoundingMethod[how.upper()]
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.round.round(
                    self.to_pylibcudf(mode="read"), decimals, plc_how
                )
            )

    def _scan(self, op: str) -> ColumnBase:
        return self.scan(op.replace("cum", ""), True)._with_type_metadata(
            self.dtype
        )

    def unary_operator(self, unaryop: str) -> ColumnBase:
        unaryop_str = unaryop.upper()
        unaryop_str = _unaryop_map.get(unaryop_str, unaryop_str)
        unaryop_enum = plc.unary.UnaryOperator[unaryop_str]
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.unary.unary_operation(
                    self.to_pylibcudf(mode="read"), unaryop_enum
                )
            )
