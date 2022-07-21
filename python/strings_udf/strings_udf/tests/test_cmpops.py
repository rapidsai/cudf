# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_eq(data, rhs):
    # tests the `==` operator in string udfs

    def func(st):
        return st == rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_ne(data, rhs):
    # tests the `!=` operator in string udfs

    def func(st):
        return st != rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_ge(data, rhs):
    # tests the `>=` operator in string udfs

    def func(st):
        return st >= rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_le(data, rhs):
    # tests the `<=` operator in string udfs

    def func(st):
        return st <= rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_gt(data, rhs):
    # tests the `>` operator in string udfs

    def func(st):
        return st > rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_lt(data, rhs):
    # tests the `<` operator in string udfs

    def func(st):
        return st < rhs

    run_udf_test(data, func, "bool")
