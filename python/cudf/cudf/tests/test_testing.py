# Copyright (c) 2020, NVIDIA CORPORATION.

import cupy
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame, Series
from cudf.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from cudf.tests.utils import assert_eq, NUMERIC_TYPES, OTHER_TYPES


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("exact", ["equiv", True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("rname", ["a", "b"])
# @pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"])
def test_basic_assert_index_equal(
    rdata,
    exact,
    check_names,
    rname,
    check_categorical,
    dtype,
):
    p_left = pd.Index([1, 2, 3], name="a", dtype=dtype)
    p_right = pd.Index(rdata, name=rname, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    kind = None
    try:
        pd.testing.assert_index_equal(
            p_left,
            p_right,
            exact=exact,
            check_names=check_names,
            check_categorical=check_categorical,
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_index_equal(
                left,
                right,
                exact=exact,
                check_names=check_names,
                check_categorical=check_categorical,
            )
    else:
        assert_index_equal(
            left,
            right,
            exact=exact,
            check_names=check_names,
            check_categorical=check_categorical,
        )
