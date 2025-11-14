# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import collections

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "orient", ["dict", "list", "split", "tight", "records", "index", "series"]
)
@pytest.mark.parametrize(
    "into", [dict, collections.OrderedDict, collections.defaultdict(list)]
)
def test_dataframe_to_dict(orient, into):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [9, 5, 3]}, index=[10, 11, 12])
    pdf = df.to_pandas()

    actual = df.to_dict(orient=orient, into=into)
    expected = pdf.to_dict(orient=orient, into=into)
    if orient == "series":
        assert actual.keys() == expected.keys()
        for key in actual.keys():
            assert_eq(expected[key], actual[key])
    else:
        assert_eq(expected, actual)
