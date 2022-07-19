# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest
from strings_udf._typing import ends_with

@pytest.mark.parametrize(
    "data", [
        ["cudf", "rapids", "AI", "gpu", "2022", "cuda"]
    ]
)
@pytest.mark.parametrize('substr', ['c', 'cu', "2", "abc"])
def test_string_udf_endswith(data, substr):
    # tests the `endswith` function in string udfs

    def func(st):
        return ends_with(st, substr)

    run_udf_test(data, func, 'bool')
