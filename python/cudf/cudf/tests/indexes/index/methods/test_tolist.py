# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

import cudf


@pytest.mark.parametrize("data", [[1, 2, 3, 1, 2, 3, 4], []])
def test_index_tolist(data, all_supported_types_as_str):
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via the `tolist()` method. Consider using "
            r"`.to_arrow().to_pylist()` to construct a Python list."
        ),
    ):
        gdi.tolist()
