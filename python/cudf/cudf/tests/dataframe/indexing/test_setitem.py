# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd

from cudf import DataFrame


def test_setitem_datetime():
    df = DataFrame()
    df["date"] = pd.date_range("20010101", "20010105").values
    assert df.date.dtype.kind == "M"
