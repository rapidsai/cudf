# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest


@pytest.mark.parametrize("data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]])
@pytest.mark.parametrize("substr", ["a", "cu", "2", "abc"])
def test_string_udf_contains(data, substr):
    # Tests contains for string UDFs

    def func(st):
        return substr in st

    run_udf_test(data, func, "bool")
