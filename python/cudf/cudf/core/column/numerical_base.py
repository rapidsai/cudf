# Copyright (c) 2018-2021, NVIDIA CORPORATION.
"""Define an interface for columns that can perform numerical operations."""

from __future__ import annotations

from numbers import Number
from typing import Sequence, Union

import numpy as np

import cudf
from cudf import _lib as libcudf
from cudf._typing import Dtype, ScalarLike
from cudf.core.column import ColumnBase


class NumericalBaseColumn(ColumnBase):
    """A column composed of numerical data.

    This class encodes a standard interface for different types of columns
    containing numerical types of data. In particular, mathematical operations
    that make sense whether a column is integral or real, fixed or floating
    point, should be encoded here.
    """

    def reduce(
        self, op: str, skipna: bool = None, min_count: int = 0, **kwargs
    ) -> ScalarLike:
        """Perform a reduction operation.

        op : str
            The operation to perform.
        skipna : bool
            Whether or not na values must be
        """
        preprocessed = self._process_for_reduction(
            skipna=skipna, min_count=min_count
        )
        if isinstance(preprocessed, ColumnBase):
            return libcudf.reduce.reduce(op, preprocessed, **kwargs)
        else:
            return preprocessed

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> ScalarLike:
        return self.reduce(
            "sum", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def product(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> ScalarLike:
        return self.reduce(
            "product", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def mean(
        self, skipna: bool = None, dtype: Dtype = np.float64
    ) -> ScalarLike:
        return self.reduce("mean", skipna=skipna, dtype=dtype)

    def var(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = np.float64
    ) -> ScalarLike:
        return self.reduce("var", skipna=skipna, dtype=dtype, ddof=ddof)

    def std(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = np.float64
    ) -> ScalarLike:
        return self.reduce("std", skipna=skipna, dtype=dtype, ddof=ddof)

    def sum_of_squares(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> ScalarLike:
        return self.reduce(
            "sum_of_squares", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def _can_return_nan(self, skipna: bool = None) -> bool:
        return not skipna and self.has_nulls()

    def kurtosis(self, skipna: bool = None) -> float:
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
        term_one_section_two = m4_numerator / (V ** 2)
        term_two = ((n - 1) ** 2) / ((n - 2) * (n - 3))
        kurt = term_one_section_one * term_one_section_two - 3 * term_two
        return kurt

    def skew(self, skipna: bool = None) -> ScalarLike:
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
        self, q: Union[float, Sequence[float]], interpolation: str, exact: bool
    ) -> NumericalBaseColumn:
        if isinstance(q, Number) or cudf.api.types.is_list_like(q):
            np_array_q = np.asarray(q)
            if np.logical_or(np_array_q < 0, np_array_q > 1).any():
                raise ValueError(
                    "percentiles should all be in the interval [0, 1]"
                )
        # Beyond this point, q either being scalar or list-like
        # will only have values in range [0, 1]
        result = self._numeric_quantile(q, interpolation, exact)
        if isinstance(q, Number):
            return (
                cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
                if result[0] is cudf.NA
                else result[0]
            )
        return result

    def median(self, skipna: bool = None) -> NumericalBaseColumn:
        skipna = True if skipna is None else skipna

        if self._can_return_nan(skipna=skipna):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        # enforce linear in case the default ever changes
        return self.quantile(0.5, interpolation="linear", exact=True)

    def _numeric_quantile(
        self, q: Union[float, Sequence[float]], interpolation: str, exact: bool
    ) -> NumericalBaseColumn:
        quant = [float(q)] if not isinstance(q, (Sequence, np.ndarray)) else q
        # get sorted indices and exclude nulls
        sorted_indices = self.as_frame()._get_sorted_inds(
            ascending=True, na_position="first"
        )
        sorted_indices = sorted_indices[self.null_count :]

        return libcudf.quantiles.quantile(
            self, quant, interpolation, sorted_indices, exact
        )

    def cov(self, other: ColumnBase) -> float:
        if (
            len(self) == 0
            or len(other) == 0
            or (len(self) == 1 and len(other) == 1)
        ):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        result = (self - self.mean()) * (other - other.mean())
        cov_sample = result.sum() / (len(self) - 1)
        return cov_sample

    def corr(self, other: ColumnBase) -> float:
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
        """Round the values in the Column to the given number of decimals."""
        return libcudf.round.round(self, decimal_places=decimals, how=how)

    def _apply_scan_op(self, op: str) -> ColumnBase:
        return libcudf.reduce.scan(op, self, True)._with_type_metadata(
            self.dtype
        )
