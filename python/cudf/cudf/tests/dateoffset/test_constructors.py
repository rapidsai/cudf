# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf


def test_dateoffset_instance_subclass_check():
    assert not issubclass(pd.DateOffset, cudf.DateOffset)
    assert not isinstance(pd.DateOffset(), cudf.DateOffset)
