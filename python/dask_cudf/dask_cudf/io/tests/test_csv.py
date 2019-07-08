import warnings

import numpy as np
import pandas as pd
import pytest

import dask
import dask.dataframe as dd

import dask_cudf


def test_read_csv(tmp_path):
    df = dask.datasets.timeseries(
        dtypes={"x": int, "y": int}, freq="120s"
    ).reset_index(drop=True)

    df.to_csv(tmp_path / "data-*.csv", index=False)

    df2 = dask_cudf.read_csv(tmp_path / "data-*.csv")
    dd.assert_eq(df, df2)

    # file path test
    stmp_path = str(tmp_path / "data-*.csv")
    df3 = dask_cudf.read_csv(f"file://{stmp_path}")
    dd.assert_eq(df2, df3)


def test_raises_FileNotFoundError():
    with pytest.raises(FileNotFoundError):
        dask_cudf.read_csv("foo.csv")


def test_read_csv_w_bytes(tmp_path):
    df = dask.datasets.timeseries(
        dtypes={"x": int, "y": int}, freq="120s"
    ).reset_index(drop=True)
    df = pd.DataFrame(dict(x=np.arange(20), y=np.arange(20)))
    df.to_csv(tmp_path / "data-*.csv", index=False)

    df2 = dask_cudf.read_csv(tmp_path / "*.csv", chunksize="50 B")
    assert df2.npartitions == 3
    dd.assert_eq(df2, df, check_index=False)


def test_read_csv_compression(tmp_path):
    df = pd.DataFrame(dict(x=np.arange(20), y=np.arange(20)))
    df.to_csv(tmp_path / "data.csv.gz", index=False, compression="gzip")

    with pytest.warns(UserWarning) as w:
        df2 = dask_cudf.read_csv(
            tmp_path / "*.csv.gz", chunksize="50 B", compression="gzip"
        )

    assert len(w) == 1
    msg = str(w[0].message)
    assert "gzip" in msg

    assert df2.npartitions == 1
    dd.assert_eq(df2, df, check_index=False)

    with warnings.catch_warnings(record=True) as record:
        df2 = dask_cudf.read_csv(
            tmp_path / "*.csv.gz", chunksize=None, compression="gzip"
        )

        assert not record
