# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import dask.dataframe as dd

import cudf


def _make_random_frame(nelem, npartitions=2, include_na=False):
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {"x": rng.random(size=nelem), "y": rng.random(size=nelem)}
    )

    if include_na:
        df["x"][::2] = pd.NA

    gdf = cudf.DataFrame(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf
