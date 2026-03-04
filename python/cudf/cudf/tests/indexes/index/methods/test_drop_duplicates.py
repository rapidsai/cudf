# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("data", [[1, 2, 3, 1, 2, 3, 4], [], [1], [1, 2, 3]])
def test_index_drop_duplicates(data, all_supported_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            len(data) > 0
            and all_supported_types_as_str
            in {"timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"},
            reason=f"wrong result for {all_supported_types_as_str}",
        )
    )
    pdi = pd.Index(data, dtype=all_supported_types_as_str)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    assert_eq(pdi.drop_duplicates(), gdi.drop_duplicates())
