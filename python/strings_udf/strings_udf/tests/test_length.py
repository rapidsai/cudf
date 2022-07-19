# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize("data", [["cudf", "rapids", "AI", "gpu", "2022"]])
def test_string_udf_len(data):
    # tests the `len` function in string udfs

    def func(st):
        return len(st)

    run_udf_test(data, func, "int64")
