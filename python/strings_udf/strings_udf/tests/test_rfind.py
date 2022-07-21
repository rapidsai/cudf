# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", "", "gpu"])
def test_string_udf_rfind(data, substr):
    # tests the `rfind` function in string udfs

    def func(st):
        return st.rfind(substr)

    run_udf_test(data, func, "int32")
