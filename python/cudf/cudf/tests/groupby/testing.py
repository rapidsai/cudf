# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd

import cudf
from cudf.testing import assert_eq


def assert_groupby_results_equal(
    expect, got, sort=True, as_index=True, by=None, **kwargs
):
    # Because we don't sort by index by default in groupby,
    # sort expect and got by index before comparing.
    if sort:
        if as_index:
            expect = expect.sort_index()
            got = got.sort_index()
        else:
            assert by is not None
            if isinstance(expect, (pd.DataFrame, cudf.DataFrame)):
                expect = expect.sort_values(by=by).reset_index(drop=True)
            else:
                expect = expect.sort_values(by=by).reset_index(drop=True)

            if isinstance(got, cudf.DataFrame):
                got = got.sort_values(by=by).reset_index(drop=True)
            else:
                got = got.sort_values(by=by).reset_index(drop=True)

    assert_eq(expect, got, **kwargs)
