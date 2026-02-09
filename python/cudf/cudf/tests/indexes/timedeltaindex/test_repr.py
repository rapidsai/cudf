# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pytest

import cudf


@pytest.mark.parametrize(
    "index, expected_repr",
    [
        (
            lambda: cudf.Index(
                [1000000, 200000, 3000000], dtype="timedelta64[ms]"
            ),
            "TimedeltaIndex(['0 days 00:16:40', "
            "'0 days 00:03:20', '0 days 00:50:00'], "
            "dtype='timedelta64[ms]')",
        ),
        (
            lambda: cudf.Index(
                [None, None, None, None, None], dtype="timedelta64[us]"
            ),
            "TimedeltaIndex([NaT, NaT, NaT, NaT, NaT], "
            "dtype='timedelta64[us]')",
        ),
        (
            lambda: cudf.Index(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[us]",
            ),
            "TimedeltaIndex([0 days 00:02:16.457654, NaT, "
            "0 days 00:04:05.345345, "
            "0 days 00:03:43.432411, NaT,"
            "       0 days 01:00:34.548734, 0 days 00:00:00.023234],"
            "      dtype='timedelta64[us]')",
        ),
        (
            lambda: cudf.Index(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[s]",
            ),
            "TimedeltaIndex([1579 days 08:54:14, NaT, 2839 days 15:29:05,"
            "       2586 days 00:33:31, NaT, 42066 days 12:52:14, "
            "0 days 06:27:14],"
            "      dtype='timedelta64[s]')",
        ),
    ],
)
def test_timedelta_index_repr(index, expected_repr):
    actual_repr = repr(index())

    assert actual_repr.split() == expected_repr.split()
