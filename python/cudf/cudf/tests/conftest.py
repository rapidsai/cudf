import pathlib

import pytest

import rmm  # noqa: F401


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"

@pytest.fixture(scope="session", autouse=True)
def patch_df_series_to_pandas():
    from cudf import DataFrame, Series
    DataFrame.to_pandas.__defaults__ = (False,)
    Series.to_pandas.__defaults__ = (True, False)
