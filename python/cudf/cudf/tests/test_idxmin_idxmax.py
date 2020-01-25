import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
def test_dataframe_idxmin(skipna, axis):
    if skipna:
        pdf = pd.DataFrame(
            {"A": [4, 5, 2, None], "B": [11, 2, None, 8], "C": [1, 8, 66, 4]}
        )
    else:
        pdf = pd.DataFrame(
            {"A": [4, 5, 2, 6], "B": [11, 2, 5, 8], "C": [1, 8, 66, 4]}
        )

    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.idxmin(skipna=skipna, axis=axis),
        gdf.idxmin(skipna=skipna, axis=axis),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
def test_dataframe_idxmax(skipna, axis):
    if True:
        pdf = pd.DataFrame(
            {"A": [4, 5, 2, None], "B": [11, 2, None, 8], "C": [1, 8, 66, 4]}
        )
    else:
        pdf = pd.DataFrame(
            {"A": [4, 5, 2, 6], "B": [11, 2, 5, 8], "C": [1, 8, 66, 4]}
        )

    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.idxmax(skipna=skipna, axis=axis),
        gdf.idxmax(skipna=skipna, axis=axis),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
def test_series_idxmin(skipna, axis):
    if True:
        psr = pd.Series(
            data=[1, None, 4, 3, 4], index=["A", "B", "C", "D", "E"]
        )
    else:
        psr = pd.Series(data=[1, 2, 4, 3, 4], index=["A", "B", "C", "D", "E"])

    gsr = cudf.from_pandas(psr)
    assert_eq(
        psr.idxmin(skipna=skipna, axis=axis),
        gsr.idxmin(skipna=skipna, axis=axis),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
def test_series_idxmax(skipna, axis):
    if True:
        psr = pd.Series(
            data=[1, None, 4, 3, 4], index=["A", "B", "C", "D", "E"]
        )
    else:
        psr = pd.Series(data=[1, 2, 4, 3, 4], index=["A", "B", "C", "D", "E"])

    psr = pd.Series(data=[1, None, 4, 3, 4], index=["A", "B", "C", "D", "E"])
    gsr = cudf.from_pandas(psr)
    assert_eq(
        psr.idxmin(skipna=skipna, axis=axis),
        gsr.idxmin(skipna=skipna, axis=axis),
    )
