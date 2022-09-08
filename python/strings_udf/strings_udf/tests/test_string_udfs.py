# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

from .utils import run_udf_test

string_data = [
    "abc",
    "ABC",
    "AbC",
    "123",
    "123aBc",
    "123@.!",
    "",
    "rapids ai",
    "gpu",
    "True",
    "False",
    "1.234",
    ".123a",
    "0.013",
    "1.0",
    "01",
    "20010101",
    "cudf",
    "cuda",
    "gpu",
]


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_eq(data, rhs):
    def func(st):
        return st == rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_ne(data, rhs):
    def func(st):
        return st != rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_ge(data, rhs):
    def func(st):
        return st >= rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_le(data, rhs):
    def func(st):
        return st <= rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_gt(data, rhs):
    def func(st):
        return st > rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("rhs", ["cudf", "cuda", "gpucudf", "abc"])
def test_string_udf_lt(data, rhs):
    def func(st):
        return st < rhs

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("substr", ["a", "cu", "2", "abc"])
def test_string_udf_contains(data, substr):
    def func(st):
        return substr in st

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", ""])
def test_string_udf_count(data, substr):
    def func(st):
        return st.count(substr)

    run_udf_test(data, func, "int32")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", "", "gpu"])
def test_string_udf_find(data, substr):
    def func(st):
        return st.find(substr)

    run_udf_test(data, func, "int32")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc"])
def test_string_udf_endswith(data, substr):
    def func(st):
        return st.endswith(substr)

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isalnum(data):
    def func(st):
        return st.isalnum()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isalpha(data):
    def func(st):
        return st.isalpha()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isdecimal(data):
    def func(st):
        return st.isdecimal()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isdigit(data):
    def func(st):
        return st.isdigit()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_islower(data):
    def func(st):
        return st.islower()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isnumeric(data):
    def func(st):
        return st.isnumeric()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isspace(data):
    def func(st):
        return st.isspace()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_isupper(data):
    def func(st):
        return st.isupper()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [string_data])
def test_string_udf_len(data):
    def func(st):
        return len(st)

    run_udf_test(data, func, "int64")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", "", "gpu"])
def test_string_udf_rfind(data, substr):
    def func(st):
        return st.rfind(substr)

    run_udf_test(data, func, "int32")


@pytest.mark.parametrize("data", [string_data])
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc"])
def test_string_udf_startswith(data, substr):
    def func(st):
        return st.startswith(substr)

    run_udf_test(data, func, "bool")
