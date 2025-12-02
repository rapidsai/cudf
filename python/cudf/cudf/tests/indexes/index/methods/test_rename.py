# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    SERIES_OR_INDEX_NAMES,
)


@pytest.mark.parametrize("initial_name", SERIES_OR_INDEX_NAMES)
@pytest.mark.parametrize("name", SERIES_OR_INDEX_NAMES)
def test_index_rename(initial_name, name):
    pds = pd.Index([1, 2, 3], name=initial_name)
    gds = cudf.Index(pds)

    assert_eq(pds, gds)

    expect = pds.rename(name)
    got = gds.rename(name)

    assert_eq(expect, got)
    """
    From here on testing recursive creation
    and if name is being handles in recursive creation.
    """
    pds = pd.Index(expect)
    gds = cudf.Index(got)

    assert_eq(pds, gds)

    pds = pd.Index(pds, name="abc")
    gds = cudf.Index(gds, name="abc")
    assert_eq(pds, gds)


def test_index_rename_inplace():
    pds = pd.Index([1, 2, 3], name="asdf")
    gds = cudf.Index(pds)

    # inplace=False should yield a shallow copy
    gds_renamed_deep = gds.rename("new_name", inplace=False)

    assert gds_renamed_deep._column.data.get_ptr(
        mode="read"
    ) == gds._column.data.get_ptr(mode="read")

    # inplace=True returns none
    expected_ptr = gds._column.data_ptr
    gds.rename("new_name", inplace=True)

    assert expected_ptr == gds._column.data.get_ptr(mode="read")


def test_index_rename_preserves_arg():
    idx1 = cudf.Index([1, 2, 3], name="orig_name")

    # this should be an entirely new object
    idx2 = idx1.rename("new_name", inplace=False)

    assert idx2.name == "new_name"
    assert idx1.name == "orig_name"

    # a new object but referencing the same data
    idx3 = cudf.Index(idx1, name="last_name")

    assert idx3.name == "last_name"
    assert idx1.name == "orig_name"
