# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc"])
def test_string_udf_endswith(data, substr):
    # tests the `endswith` function in string udfs

    def func(st):
        return st.endswith(substr)

    run_udf_test(data, func, "bool")
