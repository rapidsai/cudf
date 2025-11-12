# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.fixture(params=[True, False])
def drop(request):
    """Param for `drop` argument"""
    return request.param


@pytest.mark.parametrize("level", [None, 0, "l0", 1, ["l0", 1]])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser", no_default])
def test_reset_index(level, drop, inplace, original_name, name):
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["l0", None]
    )
    ps = pd.Series(range(4), index=midx, name=original_name)
    gs = cudf.from_pandas(ps)

    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    expect = ps.reset_index(level=level, drop=drop, name=name, inplace=inplace)

    got = gs.reset_index(level=level, drop=drop, name=name, inplace=inplace)
    if inplace:
        expect = ps
        got = gs

    assert_eq(expect, got)


@pytest.mark.parametrize("level", [None, 0, 1, [None]])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index_dup_level_name(level, drop, inplace, original_name, name):
    # midx levels are named [None, None]
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    ps = pd.Series(range(4), index=midx, name=original_name)
    gs = cudf.from_pandas(ps)
    if level == [None] or not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    expect = ps.reset_index(level=level, drop=drop, inplace=inplace, name=name)
    got = gs.reset_index(level=level, drop=drop, inplace=inplace, name=name)
    if inplace:
        expect = ps
        got = gs

    assert_eq(expect, got)


@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index_named(drop, inplace, original_name, name):
    ps = pd.Series(range(4), index=["x", "y", "z", "w"], name=original_name)
    gs = cudf.from_pandas(ps)

    ps.index.name = "cudf"
    gs.index.name = "cudf"

    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    expect = ps.reset_index(drop=drop, inplace=inplace, name=name)
    got = gs.reset_index(drop=drop, inplace=inplace, name=name)

    if inplace:
        expect = ps
        got = gs

    assert_eq(expect, got)


def test_reset_index_dup_level_name_exceptions():
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    ps = pd.Series(range(4), index=midx)
    gs = cudf.from_pandas(ps)

    # Should specify duplicate level names with level number.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"level": [None]},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"level": [None]},
        ),
    )

    # Cannot use drop=False and inplace=True to turn a series into dataframe.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"drop": False, "inplace": True},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"drop": False, "inplace": True},
        ),
    )

    # Pandas raises the above exception should these two inputs crosses.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"level": [None], "drop": False, "inplace": True},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"level": [None], "drop": False, "inplace": True},
        ),
    )
