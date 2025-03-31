# Copyright (c) 2021-2025, NVIDIA CORPORATION.

import datetime

import pandas as pd
import pytest

import cudf


def test_construct_timezone_scalar_error():
    pd_scalar = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(NotImplementedError):
        cudf.utils.dtypes.to_cudf_compatible_scalar(pd_scalar)

    date_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(NotImplementedError):
        cudf.utils.dtypes.to_cudf_compatible_scalar(date_scalar)
