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
    df1.to_json(tmp_path / "data-*.json")
    df2 = dask_cudf.read_json(tmp_path / "data-*.json")
    dd.assert_eq(df1, df2)


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
