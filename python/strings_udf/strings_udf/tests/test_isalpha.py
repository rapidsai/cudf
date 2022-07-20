# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest
from strings_udf._typing import isalpha

@pytest.mark.parametrize("data", [["abc", "1@2", "123abc", "2.1", "@Aa", "ABC"]])
def test_string_udf_isalpha(data):
    # tests the `isalpha` function in string udfs

    def func(st):
        return isalpha(st)

    run_udf_test(data, func, 'bool')
