# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal


@pytest.mark.parametrize(
    "by,data",
    [
        ([], []),
        ([1, 1, 2, 2], [0, 0, 1, 1]),
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([1, 2, 1, 2], [0, 1, 1, 1]),
    ],
)
def test_groupby_unique(by, data, all_supported_types_as_str, request):
    pdf = pd.DataFrame({"by": by, "data": data})
    pdf["data"] = pdf["data"].astype(all_supported_types_as_str)
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby("by")["data"].unique()
    got = gdf.groupby("by")["data"].unique()
    request.applymarker(
        pytest.mark.xfail(
            len(by) == 0 and all_supported_types_as_str == "category",
            reason="pandas returns Categorical, cuDF returns np.ndarray",
        )
    )
    assert_groupby_results_equal(expect, got, check_dtype=len(by) > 0)
