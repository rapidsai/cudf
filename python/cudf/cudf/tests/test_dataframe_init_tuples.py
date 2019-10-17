# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq


def test_init_with_tuples():

    data = [
        ("a", "cat", 5, np.nan),
        ("b", None, 3, -8.4),
        ("c", "mouse", 2, 1.53),
        (None, "dog", 9, 3.2),
        ("e", "plane", 32, -5.6, "extra_parameter"),
        ("f", "bird", 2, None),
    ]

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)
