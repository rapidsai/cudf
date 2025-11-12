# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa

import cudf


def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
    assert pa_cat.equals(gd_cat.to_arrow())
