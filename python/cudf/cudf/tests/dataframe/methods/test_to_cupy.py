# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import expand_bits_to_bytes, random_bitmask


def test_empty_dataframe_to_cupy():
    df = cudf.DataFrame()

    # Check fully empty dataframe.
    mat = df.to_cupy()
    assert mat.shape == (0, 0)
    mat = df.to_numpy()
    assert mat.shape == (0, 0)

    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": rng.random(10),
            "b": rng.random(10),
            "c": rng.random(10),
        }
    )

    # Check all columns in empty dataframe.
    mat = df.head(0).to_cupy()
    assert mat.shape == (0, 3)


def test_dataframe_to_cupy():
    nelem = 123
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": rng.random(nelem),
            "b": rng.random(nelem),
            "c": rng.random(nelem),
            "d": rng.random(nelem),
        }
    )

    # Check all columns
    mat = df.to_cupy()
    assert mat.shape == (nelem, 4)
    assert mat.strides == (8, 984)

    mat = df.to_numpy()
    assert mat.shape == (nelem, 4)
    assert mat.strides == (8, 984)
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(df[k].to_numpy(), mat[:, i])

    # Check column subset
    mat = df[["a", "c"]].to_cupy().get()
    assert mat.shape == (nelem, 2)

    for i, k in enumerate("ac"):
        np.testing.assert_array_equal(df[k].to_numpy(), mat[:, i])


@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("use_na_value", [False, True])
def test_dataframe_to_cupy_single_column(has_nulls, use_na_value):
    nelem = 10
    data = np.arange(nelem, dtype=np.float64)

    if has_nulls:
        data = data.astype("object")
        data[::2] = None

    df = cudf.DataFrame({"a": data})

    if has_nulls and not use_na_value:
        with pytest.raises(ValueError, match="Column must have no nulls"):
            df.to_cupy()
        return

    na_value = 0.0 if use_na_value else None
    expected = (
        cp.asarray(df["a"].fillna(na_value))
        if has_nulls
        else cp.asarray(df["a"])
    )
    result = df.to_cupy(na_value=na_value)
    assert result.shape == (nelem, 1)
    assert_eq(result.ravel(), expected)


def test_dataframe_to_cupy_null_values():
    df = cudf.DataFrame()

    nelem = 123
    na = -10000

    refvalues = {}
    rng = np.random.default_rng(seed=0)
    for k in "abcd":
        df[k] = data = rng.random(nelem)
        bitmask = random_bitmask(nelem)
        df[k] = df[k]._column.set_mask(bitmask)
        boolmask = np.asarray(
            expand_bits_to_bytes(bitmask)[:nelem], dtype=np.bool_
        )
        data[~boolmask] = na
        refvalues[k] = data

    # Check null value causes error
    with pytest.raises(ValueError):
        df.to_cupy()
    with pytest.raises(ValueError):
        df.to_numpy()

    for k in df.columns:
        df[k] = df[k].fillna(na)

    mat = df.to_numpy()
    for i, k in enumerate(df.columns):
        np.testing.assert_array_equal(refvalues[k], mat[:, i])


@pytest.mark.parametrize("method", ["to_cupy", "to_numpy"])
@pytest.mark.parametrize("value", [1, True, 1.5])
@pytest.mark.parametrize("constructor", ["DataFrame", "Series"])
def test_to_array_categorical(method, value, constructor):
    data = [value]
    expected = getattr(pd, constructor)(data, dtype="category").to_numpy()
    result = getattr(
        getattr(cudf, constructor)(data, dtype="category"), method
    )()
    assert_eq(result, expected)
