import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.fixture
def pdf():
    return pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": [1, 2, 3, 4, 5, 6],
        }
    )


@pytest.mark.parametrize(
    "grouper",
    [
        "a",
        ["a"],
        ["a", "b"],
        np.array([0, 1, 1, 2, 3, 2]),
        {0: "a", 1: "a", 2: "b", 3: "a", 4: "b", 5: "c"},
        lambda x: x + 1,
    ],
)
def test_groupby_groups(pdf, grouper):
    gdf = cudf.from_pandas(pdf)

    for pdf_group, gdf_group in zip(
        pdf.groupby(grouper), gdf.groupby(grouper, method="libxx")
    ):
        assert pdf_group[0] == gdf_group[0]
        assert_eq(pdf_group[1], gdf_group[1])
