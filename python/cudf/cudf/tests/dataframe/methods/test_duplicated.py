# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        {
            "brand": ["Yum Yum", "Yum Yum", "Indomie", "Indomie", "Indomie"],
            "style": ["cup", "cup", "cup", "pack", "pack"],
            "rating": [4, 4, 3.5, 15, 5],
        },
        {
            "brand": ["Indomie", "Yum Yum", "Indomie", "Indomie", "Indomie"],
            "style": ["cup", "cup", "cup", "cup", "pack"],
            "rating": [4, 4, 3.5, 4, 5],
        },
    ],
)
@pytest.mark.parametrize(
    "subset", [None, ["brand"], ["rating"], ["style", "rating"]]
)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_dataframe_duplicated(data, subset, keep):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = pdf.duplicated(subset=subset, keep=keep)
    actual = gdf.duplicated(subset=subset, keep=keep)

    assert_eq(expected, actual)
