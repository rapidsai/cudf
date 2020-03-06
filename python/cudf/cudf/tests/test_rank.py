# Copyright (c) 2020, NVIDIA CORPORATION.
import numpy as np
import pandas as pd

from cudf.core import DataFrame
from cudf.tests.utils import assert_eq


def test_series_rank():
    pdf = pd.DataFrame()
    pdf["col1"] = np.array([5, 4, 3, 5, 8, 5, 2, 1, 6, 6])
    pdf["col2"] = np.array([5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf])
    gdf = DataFrame.from_pandas(pdf)

    def _check(gs, ps, method, na_option, ascending, pct):
        ranked_gs = gs.rank(
            method=method, na_option=na_option, ascending=ascending, pct=pct
        )
        ranked_ps = ps.rank(
            method=method, na_option=na_option, ascending=ascending, pct=pct
        )
        if not ranked_ps.equals(ranked_gs.to_pandas()):
            print(
                pd.concat(
                    [ps, ranked_ps, gs.to_pandas(), ranked_gs.to_pandas()],
                    axis=1,
                )
            )
        assert_eq(ranked_ps, ranked_gs.to_pandas())

    for ascending in [True, False]:
        for method in ["average", "min", "max", "first", "dense"]:
            for na_option in ["keep", "top", "bottom"]:
                for pct in [True, False]:
                    print(
                        "method=",
                        method,
                        "na_option=",
                        na_option,
                        "ascending=",
                        ascending,
                        "pct=",
                        pct,
                    )
                    # Series
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
                    # Dataframe (possible bug in pandas?)
                    # _check(gdf, pdf,
                    #  ascending=ascending, method=method, na_option=na_option)
