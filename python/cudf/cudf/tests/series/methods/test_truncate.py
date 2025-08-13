# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


def test_series_truncate():
    csr = cudf.Series([1, 2, 3, 4])
    psr = csr.to_pandas()

    assert_eq(csr.truncate(), psr.truncate())
    assert_eq(csr.truncate(1, 2), psr.truncate(1, 2))
    assert_eq(csr.truncate(before=1, after=2), psr.truncate(before=1, after=2))


def test_series_truncate_errors():
    csr = cudf.Series([1, 2, 3, 4])
    with pytest.raises(ValueError):
        csr.truncate(axis=1)
    with pytest.raises(ValueError):
        csr.truncate(copy=False)

    csr.index = [3, 2, 1, 6]
    psr = csr.to_pandas()
    assert_exceptions_equal(
        lfunc=csr.truncate,
        rfunc=psr.truncate,
    )


def test_series_truncate_datetimeindex():
    dates = cudf.date_range(
        "2021-01-01 23:45:00", "2021-01-02 23:46:00", freq="s"
    )
    csr = cudf.Series(range(len(dates)), index=dates)
    psr = csr.to_pandas()

    assert_eq(
        csr.truncate(
            before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ),
        psr.truncate(
            before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ),
    )
