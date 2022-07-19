# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest
from strings_udf._typing import find

@pytest.mark.parametrize(
    "data", [
        ["cudf", "rapids", "AI", "gpu", "2022", "cuda"]
    ]
)
@pytest.mark.parametrize('substr', ['c', 'cu', "2", "abc", "", "gpu"])
def test_string_udf_find(data, substr):
    # tests the `find` function in string udfs

    def func(st):
        return find(st, substr)

    run_udf_test(data, func, 'int32')
