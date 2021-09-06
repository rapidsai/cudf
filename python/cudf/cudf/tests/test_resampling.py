import pandas as pd

import cudf
from cudf.testing._utils import assert_eq


def assert_resample_results_equal(lhs, rhs, **kwargs):
    assert_eq(
        lhs.sort_index(),
        rhs.sort_index(),
        check_dtype=False,
        check_freq=False,
        **kwargs,
    )


def test_series_downsample_simple():
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1T")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    assert_resample_results_equal(
        psr.resample("3T").sum(), gsr.resample("3T").sum(),
    )
