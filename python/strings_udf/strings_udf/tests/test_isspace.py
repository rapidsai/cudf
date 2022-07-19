# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest
from strings_udf._typing import isspace

@pytest.mark.parametrize("data", [["1", "   x   ", " ", "2.1", "", "0003"]])
def test_string_udf_isspace(data):
    # tests the `isspace` function in string udfs

    def func(st):
        return isspace(st)

    run_udf_test(data, func, 'bool')
