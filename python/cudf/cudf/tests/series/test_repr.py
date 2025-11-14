# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import textwrap

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import cudf


@pytest.mark.parametrize("nrows", [0, 5, 10])
def test_null_series(nrows, all_supported_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str in {"bool", "timedelta64[ms]"},
            reason=f"cuDF repr doesn't match pandas repr for {all_supported_types_as_str}",
        )
    )
    rng = np.random.default_rng(seed=0)
    size = 5
    sr = cudf.Series(rng.integers(1, 9, size)).astype(
        all_supported_types_as_str
    )
    sr[rng.choice([False, True], size=size)] = None
    if all_supported_types_as_str != "category" and cudf.dtype(
        all_supported_types_as_str
    ).kind in {"u", "i"}:
        ps = sr.to_pandas(nullable=True)
    else:
        ps = sr.to_pandas()

    with pd.option_context("display.max_rows", int(nrows)):
        psrepr = repr(ps).replace("NaN", "<NA>").replace("None", "<NA>")
        if "UInt" in psrepr:
            psrepr = psrepr.replace("UInt", "uint")
        elif "Int" in psrepr:
            psrepr = psrepr.replace("Int", "int")
        assert psrepr.split() == repr(sr).split()


@pytest.mark.parametrize("nrows", [None, 0, 2, 10, 20, 21])
def test_full_series(nrows, all_supported_types_as_str):
    size = 20
    rng = np.random.default_rng(seed=0)
    ps = pd.Series(rng.integers(0, 100, size)).astype(
        all_supported_types_as_str
    )
    sr = cudf.from_pandas(ps)
    with pd.option_context("display.max_rows", nrows):
        assert repr(ps) == repr(sr)


@given(
    st.lists(
        st.integers(-9223372036854775808, 9223372036854775807), max_size=1000
    )
)
@settings(deadline=None, max_examples=20)
def test_integer_series(x):
    sr = cudf.Series(x, dtype=int)
    ps = pd.Series(data=x, dtype=int)

    assert repr(sr) == repr(ps)


@given(st.lists(st.floats()))
@settings(deadline=None, max_examples=20)
def test_float_series(x):
    sr = cudf.Series(x, dtype=float, nan_as_null=False)
    ps = pd.Series(data=x, dtype=float)
    assert repr(sr) == repr(ps)


@pytest.mark.parametrize(
    "sr,pandas_special_case",
    [
        (pd.Series([1, 2, 3], index=[10, 20, None]), False),
        (pd.Series([1, None, 3], name="a", index=[None, "a", "b"]), True),
        (pd.Series(None, index=[None, "a", "b"], dtype="float"), True),
        (pd.Series([None, None], name="aa", index=[None, None]), False),
        (pd.Series([1, 2, 3], index=[None, None, None]), False),
        (
            pd.Series(
                [None, 2, 3],
                index=np.array([1, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.Series(
                [None, None, None],
                index=np.array([None, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.Series(
                [1, None, 3],
                index=np.array([10, 15, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.DataFrame(
                {"a": [1, 2, None], "v": [10, None, 22], "p": [100, 200, 300]}
            ).set_index(["a", "v"])["p"],
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": [1, 2, None],
                    "v": ["n", "c", "a"],
                    "p": [None, None, None],
                }
            ).set_index(["a", "v"])["p"],
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": np.array([1, None, None], dtype="datetime64[ns]"),
                    "v": ["n", "c", "a"],
                    "p": [None, None, None],
                }
            ).set_index(["a", "v"])["p"],
            False,
        ),
    ],
)
def test_series_null_index_repr(sr, pandas_special_case):
    psr = sr
    gsr = cudf.from_pandas(psr)

    expected_repr = repr(psr).replace("NaN", "<NA>").replace("None", "<NA>")
    actual_repr = repr(gsr)

    if pandas_special_case:
        # Pandas inconsistently print Index null values
        # as `None` at some places and `NaN` at few other places
        # Whereas cudf is consistent with strings `null` values
        # to be printed as `None` everywhere.
        actual_repr = repr(gsr).replace("None", "<NA>")
    assert expected_repr.split() == actual_repr.split()


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [
            136457654,
            134736784,
            245345345,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("dtype", ["timedelta64[s]", "timedelta64[us]"])
def test_timedelta_series_s_us_repr(data, dtype):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = repr(psr).replace("timedelta64[ns]", dtype)
    actual = repr(sr)

    assert expected.split() == actual.split()


@pytest.mark.parametrize(
    "ser, expected_repr",
    [
        (
            lambda: cudf.Series([], dtype="timedelta64[ns]"),
            textwrap.dedent(
                """
            Series([], dtype: timedelta64[ns])
            """
            ),
        ),
        (
            lambda: cudf.Series([], dtype="timedelta64[ms]"),
            textwrap.dedent(
                """
            Series([], dtype: timedelta64[ms])
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [1000000, 200000, 3000000], dtype="timedelta64[ns]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.001000
            1    0 days 00:00:00.000200
            2    0 days 00:00:00.003000
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [1000000, 200000, 3000000], dtype="timedelta64[ms]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:16:40
            1    0 days 00:03:20
            2    0 days 00:50:00
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [1000000, 200000, None], dtype="timedelta64[ns]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.001000000
            1    0 days 00:00:00.000200000
            2                          NaT
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [1000000, 200000, None], dtype="timedelta64[ms]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:16:40
            1    0 days 00:03:20
            2                NaT
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [None, None, None, None, None], dtype="timedelta64[ns]"
            ),
            textwrap.dedent(
                """
            0    NaT
            1    NaT
            2    NaT
            3    NaT
            4    NaT
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [None, None, None, None, None], dtype="timedelta64[ms]"
            ),
            textwrap.dedent(
                """
            0    NaT
            1    NaT
            2    NaT
            3    NaT
            4    NaT
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [12, 12, 22, 343, 4353534, 435342], dtype="timedelta64[ns]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.000000012
            1    0 days 00:00:00.000000012
            2    0 days 00:00:00.000000022
            3    0 days 00:00:00.000000343
            4    0 days 00:00:00.004353534
            5    0 days 00:00:00.000435342
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [12, 12, 22, 343, 4353534, 435342], dtype="timedelta64[ms]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.012000
            1    0 days 00:00:00.012000
            2    0 days 00:00:00.022000
            3    0 days 00:00:00.343000
            4    0 days 01:12:33.534000
            5    0 days 00:07:15.342000
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
                dtype="timedelta64[ns]",
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.000000001
            1    0 days 00:00:00.000001132
            2    0 days 00:00:00.023223231
            3    0 days 00:00:00.000000233
            4              0 days 00:00:00
            5    0 days 00:00:00.000000332
            6    0 days 00:00:00.000000323
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
                dtype="timedelta64[ms]",
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.001000
            1    0 days 00:00:01.132000
            2    0 days 06:27:03.231000
            3    0 days 00:00:00.233000
            4           0 days 00:00:00
            5    0 days 00:00:00.332000
            6    0 days 00:00:00.323000
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ms]",
            ),
            textwrap.dedent(
                """
            0    157937 days 02:23:52.432000
            1         1 days 13:25:36.784000
            2         2 days 20:09:05.345000
            3         2 days 14:03:52.411000
            4     11573 days 23:39:03.241000
            5        42 days 01:35:48.734000
            6         0 days 00:00:23.234000
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ns]",
            ),
            textwrap.dedent(
                """
            0    0 days 03:47:25.765432432
            1    0 days 00:00:00.134736784
            2    0 days 00:00:00.245345345
            3    0 days 00:00:00.223432411
            4    0 days 00:16:39.992343241
            5    0 days 00:00:03.634548734
            6    0 days 00:00:00.000023234
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ms]",
                name="abc",
            ),
            textwrap.dedent(
                """
            0    157937 days 02:23:52.432000
            1         1 days 13:25:36.784000
            2         2 days 20:09:05.345000
            3         2 days 14:03:52.411000
            4     11573 days 23:39:03.241000
            5        42 days 01:35:48.734000
            6         0 days 00:00:23.234000
            Name: abc, dtype: timedelta64[ms]
            """
            ),
        ),
        (
            lambda: cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ns]",
                index=["a", "b", "z", "x", "y", "l", "m"],
                name="hello",
            ),
            textwrap.dedent(
                """
            a    0 days 03:47:25.765432432
            b    0 days 00:00:00.134736784
            z    0 days 00:00:00.245345345
            x    0 days 00:00:00.223432411
            y    0 days 00:16:39.992343241
            l    0 days 00:00:03.634548734
            m    0 days 00:00:00.000023234
            Name: hello, dtype: timedelta64[ns]
            """
            ),
        ),
    ],
)
def test_timedelta_series_ns_ms_repr(ser, expected_repr):
    expected = expected_repr
    actual = repr(ser())

    assert expected.split() == actual.split()


def test_categorical_series_with_nan_repr():
    series = cudf.Series(
        [1, 2, np.nan, 10, np.nan, None], nan_as_null=False
    ).astype("category")

    expected_repr = textwrap.dedent(
        """
    0     1.0
    1     2.0
    2     NaN
    3    10.0
    4     NaN
    5    <NA>
    dtype: category
    Categories (4, float64): [1.0, 2.0, 10.0, NaN]
    """
    )

    assert repr(series).split() == expected_repr.split()

    sliced_expected_repr = textwrap.dedent(
        """
        2     NaN
        3    10.0
        4     NaN
        5    <NA>
        dtype: category
        Categories (4, float64): [1.0, 2.0, 10.0, NaN]
        """
    )

    assert repr(series[2:]).split() == sliced_expected_repr.split()


def test_empty_series_name():
    ps = pd.Series([], name="abc", dtype="int")
    gs = cudf.from_pandas(ps)

    assert repr(ps) == repr(gs)


@pytest.mark.parametrize("item", [0, slice(0, 1)])
@pytest.mark.parametrize("data", [["a"], ["a", None], [None]])
def test_string_repr(data, item, request):
    if data == [None]:
        request.applymarker(
            pytest.mark.xfail(
                reason="Missing value repr should be <NA> instead of None",
            )
        )
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")

    got_out = gs.iloc[item]
    expect_out = ps.iloc[item]

    expect = str(expect_out)
    got = str(got_out)
    assert expect == got
