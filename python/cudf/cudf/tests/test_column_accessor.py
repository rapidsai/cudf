import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.column_accessor import ColumnAccessor
from cudf.tests.utils import assert_eq

test_data = [
    {},
    {"a": 1},
    {"a": []},
    {"a": [1]},
    {"a": ["a"]},
    {"a": [1, 2, 3], "b": ["a", "b", "c"]},
    {("a", "b"): [1, 2, 3], ("b", "c"): [2, 3, 4]},
]


@pytest.fixture(params=test_data)
def data(request):
    return request.param


def test_iter(data):
    """
    Test that iterating over the CA
    yields column names.
    """
    ca = ColumnAccessor(data)
    for expect_key, got_key in zip(data, ca):
        assert expect_key == got_key


def test_all_columns(data):
    """
    Test that all values of the CA are
    columns.
    """
    ca = ColumnAccessor(data)
    for col in ca.values():
        assert isinstance(col, cudf.core.column.ColumnBase)


def test_column_size_mismatch():
    """
    Test that constructing a CA from columns of
    differing sizes throws an error.
    """
    with pytest.raises(ValueError):
        _ = ColumnAccessor({"a": [1], "b": [1, 2]})
