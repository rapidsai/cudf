# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_column_memory_eq,
    assert_column_memory_ne,
)


def test_multiindex_copy_sem():
    gmi = pd.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019]],
    )
    pmi = cudf.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019]],
    )
    names = ["x", "y"]
    gmi_copy = gmi.copy(names=names)
    pmi_copy = pmi.copy(names=names)
    assert_eq(gmi_copy, pmi_copy)


@pytest.mark.parametrize(
    "data",
    [
        [
            [
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
            ],
            [
                "AMZN",
                "AMZN",
                "AMZN",
                "MSFT",
                "MSFT",
                "MSFT",
                "NVDA",
                "NVDA",
                "NVDA",
            ],
        ],
        pd.MultiIndex(
            levels=[[1001, 1002], [2001, 2002]],
            codes=[[1, 1, 0, 0], [0, 1, 0, 1]],
            names=["col1", "col2"],
        ),
    ],
)
@pytest.mark.parametrize("copy_on_write", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_multiindex_copy_deep(data, copy_on_write, deep):
    """Test memory identity for deep copy
    Case1: Constructed from arrays, StringColumns
    Case2: Constructed from MultiIndex, NumericColumns
    """
    with cudf.option_context("copy_on_write", copy_on_write):
        if isinstance(data, list):
            mi1 = cudf.MultiIndex.from_arrays(data)
            mi2 = mi1.copy(deep=deep)
            for col1, col2 in zip(mi1._columns, mi2._columns, strict=True):
                if not deep or (copy_on_write and not deep):
                    assert_column_memory_eq(col1, col2)
                else:
                    assert_column_memory_ne(col1, col2)
        elif isinstance(data, pd.MultiIndex):
            data = cudf.MultiIndex(
                levels=data.levels,
                codes=data.codes,
                names=data.names,
            )
            same_ref = (not deep) or (
                cudf.get_option("copy_on_write") and not deep
            )
            mi1 = data
            mi2 = mi1.copy(deep=deep)

            # Assert ._levels identity
            lptrs = [
                lv._column.base_data.get_ptr(mode="read") for lv in mi1._levels
            ]
            rptrs = [
                lv._column.base_data.get_ptr(mode="read") for lv in mi2._levels
            ]

            assert all(
                (x == y) == same_ref for x, y in zip(lptrs, rptrs, strict=True)
            )

            # Assert ._codes identity
            lptrs = [c.base_data.get_ptr(mode="read") for c in mi1._codes]
            rptrs = [c.base_data.get_ptr(mode="read") for c in mi2._codes]

            assert all(
                (x == y) == same_ref for x, y in zip(lptrs, rptrs, strict=True)
            )

            # Assert ._data identity
            lptrs = [d.base_data.get_ptr(mode="read") for d in mi1._columns]
            rptrs = [d.base_data.get_ptr(mode="read") for d in mi2._columns]

            assert all(
                (x == y) == same_ref for x, y in zip(lptrs, rptrs, strict=True)
            )
