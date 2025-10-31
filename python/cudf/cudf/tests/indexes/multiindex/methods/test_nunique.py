# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize("array", [[1, 2], [1, None], [None, None]])
@pytest.mark.parametrize("dropna", [True, False])
def test_nunique(array, dropna):
    arrays = [array, [3, 4]]
    gidx = cudf.MultiIndex.from_arrays(arrays)
    pidx = pd.MultiIndex.from_arrays(arrays)
    result = gidx.nunique(dropna=dropna)
    expected = pidx.nunique(dropna=dropna)
    assert result == expected
