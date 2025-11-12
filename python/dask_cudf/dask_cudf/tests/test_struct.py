# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf

import dask_cudf


@pytest.mark.parametrize(
    "data, column",
    [
        (
            {
                "a": [{"a": [1, 2, 3, 4], "b": "Hello world"}, {}, {"a": []}],
                "b": [1, 2, 3],
                "c": ["rapids", "cudf", "hi"],
            },
            "a",
        ),
        (
            {"a": [{}, {}, {}], "b": [1, 2, 3], "c": ["rapids", "cudf", "hi"]},
            "a",
        ),
        (
            {
                "a": [{}, {}, {}],
                "b": [{"a": 1}, {"b": 5}, {"c": "Hello"}],
                "c": ["rapids", "cudf", "hi"],
            },
            "b",
        ),
        (
            {
                "a": [{}, {}, {}, None],
                "b": [{"a": 1}, {"b": 5}, {"c": "Hello"}, None],
                "c": ["rapids", "cudf", "hi", "cool"],
            },
            "b",
        ),
        (
            {
                "a": [{}, {}, {}, None, {}, {"a": 5}],
                "b": [
                    {"a": 1},
                    {"b": 5},
                    {"c": "Hello"},
                    None,
                    {"a": 10, "b": 5},
                    {},
                ],
                "c": ["rapids", "cudf", "hi", "cool", "hello", "world"],
            },
            "b",
        ),
    ],
)
def test_select_struct(data, column):
    df = cudf.DataFrame(data)
    ddf = dask_cudf.from_cudf(df, 2)
    assert df[column].to_arrow() == ddf[column].compute().to_arrow()
