# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

from cudf.core import DataFrame
from cudf.tests.utils import assert_eq


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
        if method == "first" and dtype == 'O':
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
            if not ranked_ps.equals(ranked_gs.to_pandas()):
                print(
                    pd.concat(
                        [ps, ranked_ps, gs.to_pandas(), ranked_gs.to_pandas()],
                        axis=1,
                    )
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
        # Dataframe (possible bug in pandas)
        # _check(
        #     gdf,
        #     pdf,
        #     method=method,
        #     na_option=na_option,
        #     ascending=ascending,
        #     pct=pct,
        # )
