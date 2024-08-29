# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import cudf
from cudf.testing import assert_eq


def test_rank_return_type_compatible_mode():
    # in compatible mode, rank() always returns floats
    df = cudf.DataFrame({"a": range(10), "b": [0] * 10}, index=[0] * 10)
    pdf = df.to_pandas()
    expect = pdf.groupby("b").get_group(0)
    result = df.groupby("b").get_group(0)
    assert_eq(expect, result)
