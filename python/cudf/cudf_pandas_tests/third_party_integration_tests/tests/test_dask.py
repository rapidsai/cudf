# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import dask.dataframe as dd
import pandas as pd


def test_sum():
    data = {"x": range(1, 11)}
    ddf = dd.from_pandas(pd.DataFrame(data), npartitions=2)
    return ddf["x"].sum().compute()
