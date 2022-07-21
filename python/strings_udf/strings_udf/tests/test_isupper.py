# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest

@pytest.mark.parametrize("data", [["1", "12", "123abc", "2.1", "", "0003", "ABC", "AbC", " 123ABC"]])
def test_string_udf_isupper(data):
    # tests the `isupper` function in string udfs

    def func(st):
        return st.isupper()

    run_udf_test(data, func, 'bool')
