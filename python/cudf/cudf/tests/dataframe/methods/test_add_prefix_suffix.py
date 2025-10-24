# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
from cudf.testing import assert_eq


def test_dataframe_add_prefix():
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_prefix("item_")
    expected = pdf.add_prefix("item_")

    assert_eq(got, expected)


def test_dataframe_add_suffix():
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_suffix("_item")
    expected = pdf.add_suffix("_item")

    assert_eq(got, expected)
