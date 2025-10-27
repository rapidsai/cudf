# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import datetime

import numpy as np
import pandas as pd

from cudf import DataFrame
from cudf.testing import assert_eq


def test_issue_165():
    df_pandas = pd.DataFrame()
    start_date = datetime.datetime.strptime("2000-10-21", "%Y-%m-%d")
    data = [(start_date + datetime.timedelta(days=x)) for x in range(6)]
    df_pandas["dates"] = data
    df_pandas["num"] = [1, 2, 3, 4, 5, 6]
    df_cudf = DataFrame(df_pandas)

    base = df_pandas.query("dates==@start_date")
    test = df_cudf.query("dates==@start_date")
    assert_eq(base, test)
    assert len(test) > 0

    mask = df_cudf.dates == start_date
    base_mask = df_pandas.dates == start_date
    assert_eq(mask, base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_ts = pd.Timestamp(start_date)
    test = df_cudf.query("dates==@start_date_ts")
    base = df_pandas.query("dates==@start_date_ts")
    assert_eq(base, test)
    assert len(test) > 0

    mask = df_cudf.dates == start_date_ts
    base_mask = df_pandas.dates == start_date_ts
    assert_eq(mask, base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_np = np.datetime64(start_date_ts, "ns")
    test = df_cudf.query("dates==@start_date_np")
    base = df_pandas.query("dates==@start_date_np")
    assert_eq(base, test)
    assert len(test) > 0

    mask = df_cudf.dates == start_date_np
    base_mask = df_pandas.dates == start_date_np
    assert_eq(mask, base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0
