# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [0, "index"])
def test_dataframe_index_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = cudf.DataFrame.from_pandas(pdf)

    expect = pdf.rename(mapper={1: 5, 2: 6}, axis=axis)
    got = gdf.rename(mapper={1: 5, 2: 6}, axis=axis)

    assert_eq(expect, got)

    expect = pdf.rename(index={1: 5, 2: 6})
    got = gdf.rename(index={1: 5, 2: 6})

    assert_eq(expect, got)

    expect = pdf.rename({1: 5, 2: 6})
    got = gdf.rename({1: 5, 2: 6})

    assert_eq(expect, got)

    # `pandas` can support indexes with mixed values. We throw a
    # `NotImplementedError`.
    with pytest.raises(NotImplementedError):
        gdf.rename(mapper={1: "x", 2: "y"}, axis=axis)


def test_dataframe_MI_rename():
    gdf = cudf.DataFrame(
        {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
    )
    gdg = gdf.groupby(["a", "b"]).count()
    pdg = gdg.to_pandas()

    expect = pdg.rename(mapper={1: 5, 2: 6}, axis=0)
    got = gdg.rename(mapper={1: 5, 2: 6}, axis=0)

    assert_eq(expect, got)


@pytest.mark.parametrize("axis", [1, "columns"])
def test_dataframe_column_rename(axis):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf = cudf.DataFrame.from_pandas(pdf)

    expect = pdf.rename(mapper=lambda name: 2 * name, axis=axis)
    got = gdf.rename(mapper=lambda name: 2 * name, axis=axis)

    assert_eq(expect, got)

    expect = pdf.rename(columns=lambda name: 2 * name)
    got = gdf.rename(columns=lambda name: 2 * name)

    assert_eq(expect, got)

    rename_mapper = {"a": "z", "b": "y", "c": "x"}
    expect = pdf.rename(columns=rename_mapper)
    got = gdf.rename(columns=rename_mapper)

    assert_eq(expect, got)
