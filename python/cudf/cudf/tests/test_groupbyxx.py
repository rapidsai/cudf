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


def test_groupby_groups(pdf):
    gdf = cudf.from_pandas(pdf)

    for pdf_group, gdf_group in zip(
        pdf.groupby("a"), gdf.groupby("a", method="libxx")
    ):
        assert pdf_group[0] == gdf_group[0]
        assert_eq(pdf_group[1], gdf_group[1])
