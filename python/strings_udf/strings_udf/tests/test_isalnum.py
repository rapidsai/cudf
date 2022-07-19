# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest
from strings_udf._typing import isalnum

@pytest.mark.parametrize("data", [["1", "1@2", "123abc", "2.1", "", "0003"]])
def test_string_udf_isalnum(data):
    # tests the `rfind` function in string udfs

    def func(st):
        return isalnum(st)

    run_udf_test(data, func, 'bool')
