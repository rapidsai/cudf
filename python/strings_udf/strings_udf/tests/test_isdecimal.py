# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest
from strings_udf._typing import isdecimal

@pytest.mark.parametrize("data", [["1", "12", "123abc", "2.1", "", "0003"]])
def test_string_udf_isdecimal(data):
    # tests the `isdecimal` function in string udfs

    def func(st):
        return isdecimal(st)

    run_udf_test(data, func, 'bool')
