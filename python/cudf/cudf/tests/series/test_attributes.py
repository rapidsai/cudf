# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import re

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_iter_error():
    gs = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.items()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.iteritems()

    with pytest.raises(TypeError):
        iter(gs._column)


@pytest.mark.parametrize("data", [[], [None, None], ["a", None]])
def test_series_size(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.size, gsr.size)


def test_set_index_unequal_length():
    s = cudf.Series(dtype="float64")
    with pytest.raises(ValueError):
        s.index = [1, 2, 3]
