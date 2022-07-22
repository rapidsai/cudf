# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize(
    "data", [["abc", "1@2", "123abc", "2.1", "@Aa", "ABC"]]
)
def test_string_udf_isalpha(data):
    # tests the `isalpha` function in string udfs

    def func(st):
        return st.isalpha()

    run_udf_test(data, func, "bool")
