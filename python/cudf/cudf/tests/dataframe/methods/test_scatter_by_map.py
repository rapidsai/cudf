# Copyright (c) 2018-2025, NVIDIA CORPORATION.


import numpy as np
import pytest

from cudf import DataFrame
from cudf.core.column import NumericalColumn


@pytest.mark.parametrize("map_size", [1, 8])
@pytest.mark.parametrize("nelem", [1, 10])
@pytest.mark.parametrize("keep", [True, False])
def test_dataframe_scatter_by_map(map_size, nelem, keep):
    strlist = ["dog", "cat", "fish", "bird", "pig", "fox", "cow", "goat"]
    rng = np.random.default_rng(seed=0)
    df = DataFrame(
        {
            "a": rng.choice(strlist[:map_size], nelem),
            "b": rng.uniform(low=0, high=map_size, size=nelem),
            "c": rng.integers(map_size, size=nelem),
        }
    )
    df["d"] = df["a"].astype("category")

    def _check_scatter_by_map(dfs, col):
        assert len(dfs) == map_size
        nrows = 0
        name = col.name
        for i, df in enumerate(dfs):
            nrows += len(df)
            if len(df) > 0:
                # Make sure the column types were preserved
                assert isinstance(df[name]._column, type(col._column))
            try:
                sr = df[name].astype(np.int32)
            except ValueError:
                sr = df[name]
            assert sr.nunique() <= 1
            if sr.nunique() == 1:
                if isinstance(df[name]._column, NumericalColumn):
                    assert sr.iloc[0] == i
        assert nrows == nelem

    with pytest.warns(UserWarning):
        _check_scatter_by_map(
            df.scatter_by_map("a", map_size, keep_index=keep), df["a"]
        )
    _check_scatter_by_map(
        df.scatter_by_map("b", map_size, keep_index=keep), df["b"]
    )
    _check_scatter_by_map(
        df.scatter_by_map("c", map_size, keep_index=keep), df["c"]
    )
    with pytest.warns(UserWarning):
        _check_scatter_by_map(
            df.scatter_by_map("d", map_size, keep_index=keep), df["d"]
        )

    if map_size == 2 and nelem == 100:
        with pytest.warns(UserWarning):
            df.scatter_by_map("a")  # Auto-detect map_size
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                df.scatter_by_map("a", map_size=1, debug=True)  # Bad map_size

    # Test Index
    df2 = df.set_index("c")
    generic_result = df2.scatter_by_map("b", map_size, keep_index=keep)
    _check_scatter_by_map(generic_result, df2["b"])
    if keep:
        for frame in generic_result:
            assert isinstance(frame.index, type(df2.index))

    # Test MultiIndex
    df2 = df.set_index(["a", "c"])
    multiindex_result = df2.scatter_by_map("b", map_size, keep_index=keep)
    _check_scatter_by_map(multiindex_result, df2["b"])
    if keep:
        for frame in multiindex_result:
            assert isinstance(frame.index, type(df2.index))


@pytest.mark.parametrize("ids", [[-1, 0, 1, 0], [0, 2, 3, 0]])
def test_dataframe_scatter_by_map_7513(ids):
    df = DataFrame({"id": ids, "val": [0, 1, 2, 3]})
    with pytest.raises(ValueError):
        df.scatter_by_map(df["id"])


def test_dataframe_scatter_by_map_empty():
    df = DataFrame({"a": [], "b": []}, dtype="float64")
    scattered = df.scatter_by_map(df["a"])
    assert len(scattered) == 0
