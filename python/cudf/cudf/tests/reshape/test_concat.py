# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, -2, 3, -4],
        [1.0, 12.221, 12.34, 13.324, 324.3242],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [-10, 11, -12, 13],
        [0.1, 0.002, 324.2332, 0.2342],
    ],
)
def test_series_concat_basic(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "123"],
        ["a"],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        ["abc", "123"],
        ["a"],
        ["+", "-", "!", "_", "="],
    ],
)
def test_series_concat_basic_str(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(["abc", "123"], index=[10, 20]),
        pd.Series(["a"], index=[10]),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        pd.Series(["abc", "123"], index=[50, 20]),
        pd.Series(["a"], index=[11]),
        pd.Series(["+", "-", "!", "_", "="], index=[12, 13, 14, 15, 16]),
    ],
)
def test_series_concat_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = cudf.from_pandas(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


def test_series_concat_error_mixed_types():
    gsr = cudf.Series([1, 2, 3, 4])
    other = cudf.Series(["a", "b", "c", "d"])

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        cudf.concat([gsr, other])

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        cudf.concat([gsr, gsr, other, gsr, other])


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([-1, 2, -3, 4], index=["a", "b", "c", "d"]),
        pd.Series(
            [1.0, 12.221, 12.34, 13.324, 324.3242],
            index=[
                "float one",
                "float two",
                "float three",
                "float four",
                "float five",
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            pd.Series([-10, 11, 12, 13], index=["a", "b", "c", "d"]),
            pd.Series([12, 14, -15, 27], index=["d", "e", "z", "x"]),
        ],
        [
            pd.Series(
                [0.1, 0.002, 324.2332, 0.2342], index=["-", "+", "%", "#"]
            ),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ]
        * 3,
    ],
)
def test_series_concat_list_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = [cudf.from_pandas(obj) for obj in others]

    expected = pd.concat([psr, *other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, *other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


def test_series_concat_existing_buffers():
    a1 = np.arange(10, dtype=np.float64)
    gs = cudf.Series(a1)

    # Add new buffer
    a2 = cudf.Series(np.arange(5))
    gs = cudf.concat([gs, a2])
    assert len(gs) == 15
    np.testing.assert_equal(gs.to_numpy(), np.hstack([a1, a2.to_numpy()]))

    # Ensure appending to previous buffer
    a3 = cudf.Series(np.arange(3))
    gs = cudf.concat([gs, a3])
    assert len(gs) == 18
    a4 = np.hstack([a1, a2.to_numpy(), a3.to_numpy()])
    np.testing.assert_equal(gs.to_numpy(), a4)

    # Appending different dtype
    a5 = cudf.Series(np.array([1, 2, 3], dtype=np.int32))
    a6 = cudf.Series(np.array([4.5, 5.5, 6.5], dtype=np.float64))
    gs = cudf.concat([a5, a6])
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a5.to_numpy(), a6.to_numpy()])
    )
    gs = cudf.concat([cudf.Series(a6), a5])
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a6.to_numpy(), a5.to_numpy()])
    )
