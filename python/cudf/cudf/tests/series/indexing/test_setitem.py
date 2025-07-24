# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    with pytest.raises(TypeError):
        gs[0:1] = "d"


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("bool_scalar", [True, False])
def test_set_bool_error(dtype, bool_scalar):
    sr = cudf.Series([1, 2, 3], dtype=dtype)
    psr = sr.to_pandas(nullable=True)

    assert_exceptions_equal(
        lfunc=sr.__setitem__,
        rfunc=psr.__setitem__,
        lfunc_args_and_kwargs=([bool_scalar],),
        rfunc_args_and_kwargs=([bool_scalar],),
    )


@pytest.mark.parametrize(
    "data", [[0, 1, 2], ["a", "b", "c"], [0.324, 32.32, 3243.23]]
)
def test_series_setitem_nat_with_non_datetimes(data):
    s = cudf.Series(data)
    with pytest.raises(TypeError):
        s[0] = cudf.NaT


def test_series_string_setitem():
    gs = cudf.Series(["abc", "def", "ghi", "xyz", "pqr"])
    ps = gs.to_pandas()

    gs[0] = "NaT"
    gs[1] = "NA"
    gs[2] = "<NA>"
    gs[3] = "NaN"

    ps[0] = "NaT"
    ps[1] = "NA"
    ps[2] = "<NA>"
    ps[3] = "NaN"

    assert_eq(gs, ps)


def test_series_error_nan_non_float_dtypes():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(TypeError):
        s[0] = np.nan

    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with pytest.raises(TypeError):
        s[0] = np.nan


def test_series_setitem_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s[0] = 10
