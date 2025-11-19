# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("data", [{"a": [1, 2]}, {"a": [1, 2], "b": [2, 3]}])
def test_groupby_nonempty_no_keys(data):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lambda: pdf.groupby([]),
        lambda: gdf.groupby([]),
    )
