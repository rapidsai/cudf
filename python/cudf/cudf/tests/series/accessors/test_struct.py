# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


def test_struct_iterate_error():
    s = cudf.Series(
        [{"f2": {"a": "sf21"}, "f1": "a"}, {"f1": "sf12", "f2": None}]
    )
    with pytest.raises(TypeError):
        iter(s.struct)


@pytest.mark.parametrize(
    "data",
    [
        [{}],
        [{"a": None}],
        [{"a": 1}],
        [{"a": "one"}],
        [{"a": 1}, {"a": 2}],
        [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}],
        [{"b": "two", "a": None}, None, {"a": "one", "b": "two"}],
    ],
)
def test_struct_field_errors(data):
    got = cudf.Series(data)

    with pytest.raises(KeyError):
        got.struct.field("notWithinFields")

    with pytest.raises(IndexError):
        got.struct.field(100)


def test_struct_explode():
    s = cudf.Series([], dtype=cudf.StructDtype({}))
    expect = cudf.DataFrame({})
    assert_eq(expect, s.struct.explode())

    s = cudf.Series(
        [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
            {"a": 4, "b": "a"},
        ]
    )
    expect = cudf.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "a"]})
    got = s.struct.explode()
    assert_eq(expect, got)

    # check that a copy was made:
    got["a"][0] = 5
    assert_eq(s.struct.explode(), expect)


@pytest.mark.parametrize(
    "key, expect", [(0, [1, 3]), (1, [2, 4]), ("a", [1, 3]), ("b", [2, 4])]
)
def test_struct_for_field(key, expect):
    sr = cudf.Series([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    expect = cudf.Series(expect)
    got = sr.struct.field(key)
    assert_eq(expect, got)
