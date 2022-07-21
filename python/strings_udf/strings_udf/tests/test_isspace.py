# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize("data", [["1", "   x   ", " ", "2.1", "", "0003"]])
def test_string_udf_isspace(data):
    # tests the `isspace` function in string udfs

    def func(st):
        return st.isspace()

    run_udf_test(data, func, "bool")
