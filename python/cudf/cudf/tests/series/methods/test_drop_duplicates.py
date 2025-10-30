# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("keep", ["first", "last", False])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4, 5, 6, 6],
        [],
        ["a", "b", "s", "sd", "a", "b"],
        pd.Series(["aaa"] * 10, dtype="object"),
    ],
)
def test_drop_duplicates_series(data, keep, ignore_index):
    pds = pd.Series(data)
    gds = cudf.from_pandas(pds)

    assert_eq(
        pds.drop_duplicates(keep=keep, ignore_index=ignore_index),
        gds.drop_duplicates(keep=keep, ignore_index=ignore_index),
    )

    pds.drop_duplicates(keep=keep, inplace=True, ignore_index=ignore_index)
    gds.drop_duplicates(keep=keep, inplace=True, ignore_index=ignore_index)
    assert_eq(pds, gds)
