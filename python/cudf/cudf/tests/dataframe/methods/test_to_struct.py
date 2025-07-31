# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_struct_with_datetime_and_timedelta(temporal_types_as_str):
    df = cudf.DataFrame(
        {
            "a": [12, 232, 2334],
            "datetime": cudf.Series(
                [23432, 3432423, 324324], dtype=temporal_types_as_str
            ),
        }
    )
    series = df.to_struct()
    a_array = np.array([12, 232, 2334])
    datetime_array = np.array([23432, 3432423, 324324]).astype(
        temporal_types_as_str
    )

    actual = series.to_pandas()
    values_list = []
    for i, val in enumerate(a_array):
        values_list.append({"a": val, "datetime": datetime_array[i]})

    expected = pd.Series(values_list)
    assert_eq(expected, actual)
