import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.column_accessor import ColumnAccessor
from cudf.tests.utils import assert_eq

simple_test_data = [
    {},
    {"a": []},
    {"a": [1]},
    {"a": ["a"]},
    {"a": [1, 2, 3], "b": ["a", "b", "c"]},
]

mi_test_data = [
    {("a", "b"): [1, 2, 4], ("a", "c"): [2, 3, 4]},
    {("a", "b"): [1, 2, 3], ("a", ""): [2, 3, 4]},
    {("a", "b"): [1, 2, 4], ("c", "d"): [2, 3, 4]},
    {("a", "b"): [1, 2, 3], ("a", "c"): [2, 3, 4], ("b", ""): [4, 5, 6]},
]


@pytest.fixture(params=simple_test_data)
def simple_data(request):
    return request.param


@pytest.fixture(params=mi_test_data)
def mi_data(request):
    return request.param


@pytest.fixture(params=simple_test_data + mi_test_data)
def all_data(request):
    return request.param


def test_to_pandas_simple(simple_data):
    """
    Test that a ColumnAccessor converts to a correct pd.Index
    """
    ca = ColumnAccessor(simple_data)
    assert_eq(ca.to_pandas_index(), pd.DataFrame(simple_data).columns)


def test_to_pandas_multiindex(mi_data):
    ca = ColumnAccessor(mi_data, multiindex=True)
    assert_eq(ca.to_pandas_index(), pd.DataFrame(mi_data).columns)


def test_iter(simple_data):
    """
    Test that iterating over the CA
    yields column names.
    """
    ca = ColumnAccessor(simple_data)
    for expect_key, got_key in zip(simple_data, ca):
        assert expect_key == got_key


def test_all_columns(simple_data):
    """
    Test that all values of the CA are
    columns.
    """
    ca = ColumnAccessor(simple_data)
    for col in ca.values():
        assert isinstance(col, cudf.core.column.ColumnBase)


def test_column_size_mismatch():
    """
    Test that constructing a CA from columns of
    differing sizes throws an error.
    """
    with pytest.raises(ValueError):
        _ = ColumnAccessor({"a": [1], "b": [1, 2]})
