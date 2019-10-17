# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq


def test_init_with_tuples():

    data = [
        (5, "cats", "jump", np.nan),  # 3 columns
        (2, "dogs", "dig", 7.5),  # 3 columns
        (3, "cows", "moo", -2.1, "occasionally"),  # 4 columns
    ]

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)
