# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import re

import pytest

import cudf


def test_series_tolist():
    gsr = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via the `tolist()` method. Consider using "
            r"`.to_arrow().to_pylist()` to construct a Python list."
        ),
    ):
        gsr.tolist()
