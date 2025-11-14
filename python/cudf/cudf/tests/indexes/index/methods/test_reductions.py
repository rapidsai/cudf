# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.min(),
        lambda x: x.max(),
        lambda x: x.any(),
        lambda x: x.all(),
    ],
)
def test_reductions(func):
    x = np.asarray([4, 5, 6, 10])
    idx = cudf.Index(np.asarray([4, 5, 6, 10]))

    assert func(x) == func(idx)
