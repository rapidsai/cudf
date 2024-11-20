# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import numba
import numpy as np
import pandas as pd
import pytest
from numba import cuda
from numba.core.typing import signature as nb_signature
from numba.types import CPointer, void

import rmm

import cudf
from cudf._lib.strings_udf import (
    column_from_udf_string_array,
    column_to_string_view_array,
)
from cudf.core.udf.strings_typing import (
    str_view_arg_handler,
    string_view,
    udf_string,
)
from cudf.core.udf.utils import _get_extensionty_size, _ptx_file
from cudf.testing import assert_eq
from cudf.testing._utils import sv_to_udf_str
from cudf.utils._numba import _CUDFNumbaConfig

_PTX_FILE = _ptx_file()


def get_kernels(func, dtype, size):
    """
    Create two kernels for testing a single scalar string function.
    The first tests the function's action on a string_view object and
    the second tests the same except using a udf_string object.
    Allocates an output vector with a dtype specified by the caller
    The returned kernels execute the input function on each data
    element of the input and returns the output into the output vector
    """

    func = cuda.jit(device=True)(func)

    if dtype == "str":
        outty = CPointer(udf_string)
    else:
        outty = numba.np.numpy_support.from_dtype(dtype)[::1]
    sig = nb_signature(void, CPointer(string_view), outty)

    @cuda.jit(sig, link=[_PTX_FILE], extensions=[str_view_arg_handler])
    def string_view_kernel(input_strings, output_col):
        id = cuda.grid(1)
        if id < size:
            st = input_strings[id]
            result = func(st)
            output_col[id] = result

    @cuda.jit(sig, link=[_PTX_FILE], extensions=[str_view_arg_handler])
    def udf_string_kernel(input_strings, output_col):
        # test the string function with a udf_string as input
        id = cuda.grid(1)
        if id < size:
            st = input_strings[id]
            st = sv_to_udf_str(st)
            result = func(st)
            output_col[id] = result

    return string_view_kernel, udf_string_kernel


def run_udf_test(data, func, dtype):
    """
    Run a test kernel on a set of input data
    Converts the input data to a cuDF column and subsequently
    to an array of cudf::string_view objects. It then creates
    a CUDA kernel using get_kernel which calls the input function,
    and then assembles the result back into a cuDF series before
    comparing it with the equivalent pandas result
    """
    if dtype == "str":
        output = rmm.DeviceBuffer(
            size=len(data) * _get_extensionty_size(udf_string)
        )
    else:
        dtype = np.dtype(dtype)
        output = cudf.core.column.column_empty(
            len(data), dtype=dtype, for_numba=True
        )

    cudf_column = cudf.core.column.as_column(data)
    str_views = column_to_string_view_array(cudf_column)
    sv_kernel, udf_str_kernel = get_kernels(func, dtype, len(data))

    expect = pd.Series(data).apply(func)
    with _CUDFNumbaConfig():
        sv_kernel.forall(len(data))(str_views, output)
    if dtype == "str":
        result = column_from_udf_string_array(output)
    else:
        result = output

    got = cudf.Series._from_column(result.astype(dtype))
    assert_eq(expect, got, check_dtype=False)
    with _CUDFNumbaConfig():
        udf_str_kernel.forall(len(data))(str_views, output)
    if dtype == "str":
        result = column_from_udf_string_array(output)
    else:
        result = output

    got = cudf.Series._from_column(result.astype(dtype))
    assert_eq(expect, got, check_dtype=False)


@pytest.fixture(scope="module")
def data():
    return [
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
        "This Is A Title",
        "This is Not a Title",
        "Neither is This a Title",
        "NoT a TiTlE",
        "123 Title Works",
    ]


@pytest.fixture(params=["cudf", "cuda", "gpucudf", "abc"])
def rhs(request):
    return request.param


@pytest.fixture(params=["c", "cu", "2", "abc", "", "gpu"])
def substr(request):
    return request.param


def test_string_udf_eq(data, rhs):
    def func(st):
        return st == rhs

    run_udf_test(data, func, "bool")


def test_string_udf_ne(data, rhs):
    def func(st):
        return st != rhs

    run_udf_test(data, func, "bool")


def test_string_udf_ge(data, rhs):
    def func(st):
        return st >= rhs

    run_udf_test(data, func, "bool")


def test_string_udf_le(data, rhs):
    def func(st):
        return st <= rhs

    run_udf_test(data, func, "bool")


def test_string_udf_gt(data, rhs):
    def func(st):
        return st > rhs

    run_udf_test(data, func, "bool")


def test_string_udf_lt(data, rhs):
    def func(st):
        return st < rhs

    run_udf_test(data, func, "bool")


def test_string_udf_contains(data, substr):
    def func(st):
        return substr in st

    run_udf_test(data, func, "bool")


def test_string_udf_count(data, substr):
    def func(st):
        return st.count(substr)

    run_udf_test(data, func, "int32")


def test_string_udf_find(data, substr):
    def func(st):
        return st.find(substr)

    run_udf_test(data, func, "int32")


def test_string_udf_endswith(data, substr):
    def func(st):
        return st.endswith(substr)

    run_udf_test(data, func, "bool")


def test_string_udf_isalnum(data):
    def func(st):
        return st.isalnum()

    run_udf_test(data, func, "bool")


def test_string_udf_isalpha(data):
    def func(st):
        return st.isalpha()

    run_udf_test(data, func, "bool")


def test_string_udf_isdecimal(data):
    def func(st):
        return st.isdecimal()

    run_udf_test(data, func, "bool")


def test_string_udf_isdigit(data):
    def func(st):
        return st.isdigit()

    run_udf_test(data, func, "bool")


def test_string_udf_islower(data):
    def func(st):
        return st.islower()

    run_udf_test(data, func, "bool")


def test_string_udf_isnumeric(data):
    def func(st):
        return st.isnumeric()

    run_udf_test(data, func, "bool")


def test_string_udf_isspace(data):
    def func(st):
        return st.isspace()

    run_udf_test(data, func, "bool")


def test_string_udf_isupper(data):
    def func(st):
        return st.isupper()

    run_udf_test(data, func, "bool")


def test_string_udf_istitle(data):
    def func(st):
        return st.istitle()

    run_udf_test(data, func, "bool")


def test_string_udf_len(data):
    def func(st):
        return len(st)

    run_udf_test(data, func, "int64")


def test_string_udf_rfind(data, substr):
    def func(st):
        return st.rfind(substr)

    run_udf_test(data, func, "int32")


def test_string_udf_startswith(data, substr):
    def func(st):
        return st.startswith(substr)

    run_udf_test(data, func, "bool")


def test_string_udf_return_string(data):
    def func(st):
        return st

    run_udf_test(data, func, "str")


@pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_strip(data, strip_char):
    def func(st):
        return st.strip(strip_char)

    run_udf_test(data, func, "str")


@pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_lstrip(data, strip_char):
    def func(st):
        return st.lstrip(strip_char)

    run_udf_test(data, func, "str")


@pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_rstrip(data, strip_char):
    def func(st):
        return st.rstrip(strip_char)

    run_udf_test(data, func, "str")


def test_string_udf_upper(data):
    def func(st):
        return st.upper()

    run_udf_test(data, func, "str")


def test_string_udf_lower(data):
    def func(st):
        return st.lower()

    run_udf_test(data, func, "str")


@pytest.mark.parametrize("concat_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_concat(data, concat_char):
    def func(st):
        return st + concat_char

    run_udf_test(data, func, "str")


@pytest.mark.parametrize("concat_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_concat_reflected(data, concat_char):
    def func(st):
        return concat_char + st

    run_udf_test(data, func, "str")


@pytest.mark.parametrize("to_replace", ["a", "1", "", "@"])
@pytest.mark.parametrize("replacement", ["a", "1", "", "@"])
def test_string_udf_replace(data, to_replace, replacement):
    def func(st):
        return st.replace(to_replace, replacement)

    run_udf_test(data, func, "str")
