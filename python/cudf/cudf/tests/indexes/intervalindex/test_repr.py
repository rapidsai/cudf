# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd

import cudf


def test_interval_index_repr():
    pi = pd.Index(
        [
            np.nan,
            pd.Interval(2.0, 3.0, closed="right"),
            pd.Interval(3.0, 4.0, closed="right"),
        ]
    )
    gi = cudf.from_pandas(pi)

    assert repr(pi) == repr(gi)
