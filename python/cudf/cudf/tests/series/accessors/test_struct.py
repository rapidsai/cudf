# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf


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
