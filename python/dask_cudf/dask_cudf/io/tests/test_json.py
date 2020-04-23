import os

import pandas as pd
import pytest

import dask
import dask.dataframe as dd
from dask.utils import tmpfile

import dask_cudf


def test_read_json(tmp_path):
    df1 = dask.datasets.timeseries(
        dtypes={"x": int, "y": int}, freq="120s"
    ).reset_index(drop=True)
    json_path = str(tmp_path / "data-*.json")
    df1.to_json(json_path)
    df2 = dask_cudf.read_json(json_path)
    dd.assert_eq(df1, df2)

    # file path test
    stmp_path = str(tmp_path / "data-*.json")
    df3 = dask_cudf.read_json(f"file://{stmp_path}")
    dd.assert_eq(df1, df3)

    # file list test
    list_paths = [
        os.path.join(tmp_path, fname) for fname in sorted(os.listdir(tmp_path))
    ]
    df4 = dask_cudf.read_json(list_paths)
    dd.assert_eq(df1, df4)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("orient", ["split", "index", "columns", "values"])
def test_read_json_basic(orient):
    df = pd.DataFrame({"x": ["a", "b", "c", "d"], "y": [1, 2, 3, 4]})
    with tmpfile("json") as f:
        df.to_json(f, orient=orient, lines=False)
        actual = dask_cudf.read_json(f, orient=orient, lines=False)
        actual_pd = pd.read_json(f, orient=orient, lines=False)
        dd.assert_eq(actual, actual_pd)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("lines", [True, False])
def test_read_json_lines(lines):
    df = pd.DataFrame({"x": ["a", "b", "c", "d"], "y": [1, 2, 3, 4]})
    with tmpfile("json") as f:
        df.to_json(f, orient="records", lines=lines)
        actual = dask_cudf.read_json(f, orient="records", lines=lines)
        actual_pd = pd.read_json(f, orient="records", lines=lines)
        dd.assert_eq(actual, actual_pd)
