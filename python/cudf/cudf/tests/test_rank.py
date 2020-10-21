# Copyright (c) 2020, NVIDIA CORPORATION.

from itertools import chain, combinations_with_replacement, product

import numpy as np
import pandas as pd
import pytest

from cudf.core import DataFrame
from cudf.tests.utils import assert_eq, assert_exceptions_equal


class TestRank:
    index = np.array([5, 4, 3, 2, 1, 6, 7, 8, 9, 10])
    col1 = np.array([5, 4, 3, 5, 8, 5, 2, 1, 6, 6])
    col2 = np.array([5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf])

    @pytest.mark.parametrize("dtype", ["O", "f8", "i4"])
    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize(
        "method", ["average", "min", "max", "first", "dense"]
    )
    @pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
    @pytest.mark.parametrize("pct", [True, False])
    def test_rank_all_arguments(
        self, dtype, ascending, method, na_option, pct
    ):
        if method == "first" and dtype == "O":
            # not supported by pandas
            return
        pdf = pd.DataFrame(index=self.index)
        pdf["col1"] = self.col1.astype(dtype)
        pdf["col2"] = self.col2.astype(dtype)
        gdf = DataFrame.from_pandas(pdf)

        def _check(gs, ps, method, na_option, ascending, pct):
            ranked_gs = gs.rank(
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
            ranked_ps = ps.rank(
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
            assert_eq(ranked_ps, ranked_gs.to_pandas())

        # # Series
        _check(
            gdf["col1"],
            pdf["col1"],
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )
        _check(
            gdf["col2"],
            pdf["col2"],
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )
        # TODO: https://github.com/pandas-dev/pandas/issues/32593
        # Dataframe (bug in pandas)
        # _check(
        #     gdf,
        #     pdf,
        #     method=method,
        #     na_option=na_option,
        #     ascending=ascending,
        #     pct=pct,
        # )

    def test_rank_error_arguments(self):
        pdf = pd.DataFrame(index=self.index)
        pdf["col1"] = self.col1
        pdf["col2"] = self.col2
        gdf = DataFrame.from_pandas(pdf)

        assert_exceptions_equal(
            lfunc=pdf["col1"].rank,
            rfunc=gdf["col1"].rank,
            lfunc_args_and_kwargs=(
                [],
                {
                    "method": "randomname",
                    "na_option": "keep",
                    "ascending": True,
                    "pct": True,
                },
            ),
            rfunc_args_and_kwargs=(
                [],
                {
                    "method": "randomname",
                    "na_option": "keep",
                    "ascending": True,
                    "pct": True,
                },
            ),
        )

        assert_exceptions_equal(
            lfunc=pdf["col1"].rank,
            rfunc=gdf["col1"].rank,
            lfunc_args_and_kwargs=(
                [],
                {
                    "method": "first",
                    "na_option": "randomname",
                    "ascending": True,
                    "pct": True,
                },
            ),
            rfunc_args_and_kwargs=(
                [],
                {
                    "method": "first",
                    "na_option": "randomname",
                    "ascending": True,
                    "pct": True,
                },
            ),
        )

    sort_group_args = [
        np.full((3,), np.nan),
        100 * np.random.random(10),
        np.full((3,), np.inf),
        np.full((3,), -np.inf),
    ]
    sort_dtype_args = [np.int32, np.float32, np.float64]
    # TODO: np.int64, disabled because of bug
    # https://github.com/pandas-dev/pandas/issues/32859

    @pytest.mark.parametrize(
        "elem,dtype",
        list(
            product(
                combinations_with_replacement(sort_group_args, 4),
                sort_dtype_args,
            )
        ),
    )
    def test_series_rank_combinations(self, elem, dtype):
        np.random.seed(0)
        gdf = DataFrame()
        gdf["a"] = aa = np.fromiter(
            chain.from_iterable(elem), np.float64
        ).astype(dtype)
        ranked_gs = gdf["a"].rank(method="first")
        df = pd.DataFrame()
        df["a"] = aa
        ranked_ps = df["a"].rank(method="first")
        # Check
        assert_eq(ranked_ps, ranked_gs.to_pandas())
