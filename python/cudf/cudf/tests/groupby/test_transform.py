# Copyright (c) 2024, NVIDIA CORPORATION.
import itertools

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(params=[False, True], ids=["no-null-keys", "null-keys"])
def keys_null(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["no-null-values", "null-values"])
def values_null(request):
    return request.param


@pytest.fixture
def df(keys_null, values_null):
    keys = ["a", "b", "a", "c", "b", "b", "c", "a"]
    r = range(len(keys))
    if keys_null:
        keys[::3] = itertools.repeat(None, len(r[::3]))
    values = list(range(len(keys)))
    if values_null:
        values[1::3] = itertools.repeat(None, len(r[1::3]))
    return cudf.DataFrame({"key": keys, "values": values})


@pytest.mark.parametrize("agg", ["cumsum", "cumprod", "max", "sum", "prod"])
def test_transform_broadcast(agg, df):
    pf = df.to_pandas()
    got = df.groupby("key").transform(agg)
    expect = pf.groupby("key").transform(agg)
    assert_eq(got, expect, check_dtype=False)


def test_transform_invalid():
    df = cudf.DataFrame({"key": [1, 1], "values": [4, 5]})
    with pytest.raises(TypeError):
        df.groupby("key").transform({"values": "cumprod"})
