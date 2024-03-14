# Copyright (c) 2023, NVIDIA CORPORATION.
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
def test_rank_return_type_compatible_mode(method):
    # in compatible mode, rank() always returns floats
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]})
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.from_pandas(pdf)
        result = df.groupby("a").rank(method=method)
    expect = pdf.groupby("a").rank(method=method)
    assert_eq(expect, result)
    assert result["b"].dtype == "float64"
