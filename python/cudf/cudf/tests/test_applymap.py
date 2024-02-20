# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import pytest

from cudf import NA, DataFrame
from cudf.core._compat import PANDAS_GE_210, PANDAS_GE_220
from cudf.testing import _utils as utils


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        {"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]},
        {"a": [1, 2, 3], "b": [True, False, True]},
        {"a": [1, NA, 2], "b": [NA, 4, NA]},
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: x + 1,
        lambda x: x - 0.5,
        lambda x: 2 if x is NA else 2 + (x + 1) / 4.1,
        lambda x: 42,
    ],
)
@pytest.mark.parametrize("na_action", [None, "ignore"])
def test_applymap_dataframe(data, func, na_action, request):
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_GE_220
            and request.node.callspec.id == "None-<lambda>2-data3",
            reason="https://github.com/pandas-dev/pandas/issues/57390",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_GE_220
            and request.node.callspec.id == "ignore-<lambda>3-data3",
            reason="https://github.com/pandas-dev/pandas/pull/57388",
        )
    )
    gdf = DataFrame(data)
    pdf = gdf.to_pandas(nullable=True)

    with utils.expect_warning_if(PANDAS_GE_210):
        expect = pdf.applymap(func, na_action=na_action)
    with pytest.warns(FutureWarning):
        got = gdf.applymap(func, na_action=na_action)

    utils.assert_eq(expect, got, check_dtype=False)


def test_applymap_raise_cases():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def f(x, some_kwarg=0):
        return x + some_kwarg

    with pytest.warns(FutureWarning):
        with pytest.raises(NotImplementedError):
            df.applymap(f, some_kwarg=1)

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            df.applymap(f, na_action="some_invalid_option")
