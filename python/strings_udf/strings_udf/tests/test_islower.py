# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize(
    "data", [["1", "12", "123abc", "2.1", "", "0003", "abc", "b a", "AbC"]]
)
def test_string_udf_islower(data):
    # tests the `islower` function in string udfs

    def func(st):
        return st.islower()

    run_udf_test(data, func, "bool")
