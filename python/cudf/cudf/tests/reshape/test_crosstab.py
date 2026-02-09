# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_crosstab_simple():
    a = np.array(
        [
            "foo",
            "foo",
            "foo",
            "foo",
            "bar",
            "bar",
            "bar",
            "bar",
            "foo",
            "foo",
            "foo",
        ],
        dtype=object,
    )
    b = np.array(
        [
            "one",
            "one",
            "one",
            "two",
            "one",
            "one",
            "one",
            "two",
            "two",
            "two",
            "one",
        ],
        dtype=object,
    )
    c = np.array(
        [
            "dull",
            "dull",
            "shiny",
            "dull",
            "dull",
            "shiny",
            "shiny",
            "dull",
            "shiny",
            "shiny",
            "shiny",
        ],
        dtype=object,
    )
    expected = pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
    actual = cudf.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
    assert_eq(expected, actual, check_dtype=False)
