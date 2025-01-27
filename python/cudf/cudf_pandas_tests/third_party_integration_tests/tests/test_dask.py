# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import pandas as pd

import dask.dataframe as dd


def test_sum():
    data = {"x": range(1, 11)}
    ddf = dd.from_pandas(pd.DataFrame(data), npartitions=2)
    return ddf["x"].sum().compute()
