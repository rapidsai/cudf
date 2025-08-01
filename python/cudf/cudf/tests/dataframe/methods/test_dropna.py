# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_datetime_dataframe():
    data = {
        "timearray": np.array(
            [0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]"
        )
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    assert_eq(pdf, gdf)

    assert_eq(pdf.dropna(), gdf.dropna())

    assert_eq(pdf.isnull(), gdf.isnull())

    data = np.array([0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]")
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps, gs)

    assert_eq(ps.dropna(), gs.dropna())

    assert_eq(ps.isnull(), gs.isnull())
