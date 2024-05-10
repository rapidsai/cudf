# Copyright (c) 2018-2024, NVIDIA CORPORATION.
"""Define an interface for columns that can perform numerical operations."""

from __future__ import annotations

from typing import Optional, cast

import numpy as np

import cudf
from cudf import _lib as libcudf
from cudf._typing import ScalarLike
from cudf.core.column import ColumnBase
from cudf.core.missing import NA
from cudf.core.mixins import Scannable


class NumericalBaseColumn(ColumnBase, Scannable):
    """A column composed of numerical data.

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

    def _can_return_nan(self, skipna: Optional[bool] = None) -> bool:
        return not skipna and self.has_nulls()

    def kurtosis(self, skipna: Optional[bool] = None) -> float:
        skipna = True if skipna is None else skipna

        if len(self) == 0 or self._can_return_nan(skipna=skipna):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        self = self.nans_to_nulls().dropna()  # type: ignore

        if len(self) < 4:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        n = len(self)
        miu = self.mean()
        m4_numerator = ((self - miu) ** self.normalize_binop_value(4)).sum()
        V = self.var()

        if V == 0:
            return 0

        term_one_section_one = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        term_one_section_two = m4_numerator / (V**2)
        term_two = ((n - 1) ** 2) / ((n - 2) * (n - 3))
        kurt = term_one_section_one * term_one_section_two - 3 * term_two
        return kurt

    def skew(self, skipna: Optional[bool] = None) -> ScalarLike:
        skipna = True if skipna is None else skipna

        if len(self) == 0 or self._can_return_nan(skipna=skipna):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        self = self.nans_to_nulls().dropna()  # type: ignore

        if len(self) < 3:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        n = len(self)
        miu = self.mean()
        m3 = (((self - miu) ** self.normalize_binop_value(3)).sum()) / n
        m2 = self.var(ddof=0)

        if m2 == 0:
            return 0

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
                cudf.core.column.column_empty(
                    row_count=len(q), dtype=self.dtype, masked=True
                ),
            )
        else:
            # get sorted indices and exclude nulls
            indices = libcudf.sort.order_by(
                [self], [True], "first", stable=True
            ).slice(self.null_count, len(self))
            result = libcudf.quantiles.quantile(
                self, q, interpolation, indices, exact
            )
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
                cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
                if scalar_result is NA
                else scalar_result
            )
        return result

    def mean(
        self,
        skipna: Optional[bool] = None,
        min_count: int = 0,
        dtype=np.float64,
    ):
        return self._reduce(
            "mean", skipna=skipna, min_count=min_count, dtype=dtype
        )

    def var(
        self,
        skipna: Optional[bool] = None,
        min_count: int = 0,
        dtype=np.float64,
        ddof=1,
    ):
        return self._reduce(
            "var", skipna=skipna, min_count=min_count, dtype=dtype, ddof=ddof
        )

    def std(
        self,
        skipna: Optional[bool] = None,
        min_count: int = 0,
        dtype=np.float64,
        ddof=1,
    ):
        return self._reduce(
            "std", skipna=skipna, min_count=min_count, dtype=dtype, ddof=ddof
        )

    def median(self, skipna: Optional[bool] = None) -> NumericalBaseColumn:
        skipna = True if skipna is None else skipna

        if self._can_return_nan(skipna=skipna):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        # enforce linear in case the default ever changes
        return self.quantile(
            np.array([0.5]),
            interpolation="linear",
            exact=True,
            return_scalar=True,
        )

    def cov(self, other: NumericalBaseColumn) -> float:
        if (
            len(self) == 0
            or len(other) == 0
            or (len(self) == 1 and len(other) == 1)
        ):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        result = (self - self.mean()) * (other - other.mean())
        cov_sample = result.sum() / (len(self) - 1)
        return cov_sample

    def corr(self, other: NumericalBaseColumn) -> float:
        if len(self) == 0 or len(other) == 0:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        cov = self.cov(other)
        lhs_std, rhs_std = self.std(), other.std()

        if not cov or lhs_std == 0 or rhs_std == 0:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
        return cov / lhs_std / rhs_std

    def round(
        self, decimals: int = 0, how: str = "half_even"
    ) -> NumericalBaseColumn:
        if not cudf.api.types.is_integer(decimals):
            raise TypeError("Values in decimals must be integers")
        """Round the values in the Column to the given number of decimals."""
        return libcudf.round.round(self, decimal_places=decimals, how=how)

    def _scan(self, op: str) -> ColumnBase:
        return libcudf.reduce.scan(
            op.replace("cum", ""), self, True
        )._with_type_metadata(self.dtype)
