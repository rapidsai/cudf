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


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["a", "cu", "2", "abc"])
def test_string_udf_contains(data, substr):
    # Tests contains for string UDFs

    def func(st):
        return substr in st

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", ""])
def test_string_udf_count(data, substr):
    # tests the `count` function in string udfs

    def func(st):
        return st.count(substr)

    run_udf_test(data, func, "int32")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", "", "gpu"])
def test_string_udf_find(data, substr):
    # tests the `find` function in string udfs

    def func(st):
        return st.find(substr)

    run_udf_test(data, func, "int32")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc"])
def test_string_udf_endswith(data, substr):
    # tests the `endswith` function in string udfs

    def func(st):
        return st.endswith(substr)

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [["1", "1@2", "123abc", "2.1", "", "0003"]])
def test_string_udf_isalnum(data):
    # tests the `rfind` function in string udfs

    def func(st):
        return st.isalnum()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["abc", "1@2", "123abc", "2.1", "@Aa", "ABC"]]
)
def test_string_udf_isalpha(data):
    # tests the `isalpha` function in string udfs

    def func(st):
        return st.isalpha()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [["1", "12", "123abc", "2.1", "", "0003"]])
def test_string_udf_isdecimal(data):
    # tests the `isdecimal` function in string udfs

    def func(st):
        return st.isdecimal()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [["1", "12", "123abc", "2.1", "", "0003"]])
def test_string_udf_isdigit(data):
    # tests the `isdigit` function in string udfs

    def func(st):
        return st.isdigit()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["1", "12", "123abc", "2.1", "", "0003", "abc", "b a", "AbC"]]
)
def test_string_udf_islower(data):
    # tests the `islower` function in string udfs

    def func(st):
        return st.islower()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [["1", "12", "123abc", "2.1", "", "0003"]])
def test_string_udf_isnumeric(data):
    # tests the `isnumeric` function in string udfs

    def func(st):
        return st.isnumeric()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [["1", "   x   ", " ", "2.1", "", "0003"]])
def test_string_udf_isspace(data):
    # tests the `isspace` function in string udfs

    def func(st):
        return st.isspace()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize(
    "data", [["1", "12", "123abc", "2.1", "", "0003", "ABC", "AbC", " 123ABC"]]
)
def test_string_udf_isupper(data):
    # tests the `isupper` function in string udfs

    def func(st):
        return st.isupper()

    run_udf_test(data, func, "bool")


@pytest.mark.parametrize("data", [["cudf", "rapids", "AI", "gpu", "2022"]])
def test_string_udf_len(data):
    # tests the `len` function in string udfs

    def func(st):
        return len(st)

    run_udf_test(data, func, "int64")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc", "", "gpu"])
def test_string_udf_rfind(data, substr):
    # tests the `rfind` function in string udfs

    def func(st):
        return st.rfind(substr)

    run_udf_test(data, func, "int32")


@pytest.mark.parametrize(
    "data", [["cudf", "rapids", "AI", "gpu", "2022", "cuda"]]
)
@pytest.mark.parametrize("substr", ["c", "cu", "2", "abc"])
def test_string_udf_startswith(data, substr):
    # tests the `startswith` function in string udfs

    def func(st):
        return st.startswith(substr)

    run_udf_test(data, func, "bool")
