# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data", [[1, 2, 3], ["ab", "cd", "e", None], range(0, 10)]
)
@pytest.mark.parametrize("data_name", [None, 1, "abc"])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("name", [None, no_default, 1, "abc"])
def test_index_to_frame(data, data_name, index, name):
    pidx = pd.Index(data, name=data_name)
    gidx = cudf.from_pandas(pidx)

    expected = pidx.to_frame(index=index, name=name)
    actual = gidx.to_frame(index=index, name=name)

    assert_eq(expected, actual)
