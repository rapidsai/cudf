# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("name", [None, False, "foo"])
def test_to_frame(name):
    gser = cudf.Series([1, 2, 3], name=name)
    pser = pd.Series([1, 2, 3], name=name)
    assert_eq(gser.to_frame(), pser.to_frame())
