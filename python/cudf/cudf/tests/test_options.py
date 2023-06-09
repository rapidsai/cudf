# Copyright (c) 2022-2023, NVIDIA CORPORATION.

import os
import random
from contextlib import redirect_stdout
from io import StringIO

import pytest

import cudf


@pytest.fixture(scope="class", autouse=False)
def empty_option_environment():
    old_option_environment = cudf.options._OPTIONS
    cudf.options._OPTIONS = {}
    yield
    cudf.options._OPTIONS = old_option_environment


@pytest.fixture(scope="function")
def odd_option(empty_option_environment):
    def validator(x):
        if not x % 2 == 1:
            raise ValueError(f"Invalid option value {x}")

    cudf.options._register_option(
        "odd_option",
        1,
        "An odd option.",
        validator,
    )
    yield
    del cudf.options._OPTIONS["odd_option"]


@pytest.fixture(scope="function")
def even_option(empty_option_environment):
    def validator(x):
        if not x % 2 == 0:
            raise ValueError(f"Invalid option value {x}")

    cudf.options._register_option(
        "even_option", 0, "An even option.", validator
    )
    yield
    del cudf.options._OPTIONS["even_option"]


@pytest.mark.usefixtures("odd_option", "even_option")
class TestCleanOptions:
    def test_option_get_set(odd_option):
        assert cudf.get_option("odd_option") == 1
        cudf.set_option("odd_option", 101)
        assert cudf.get_option("odd_option") == 101

    def test_option_set_invalid(odd_option):
        with pytest.raises(ValueError, match="Invalid option value 0"):
            cudf.set_option("odd_option", 0)

    def test_option_description(odd_option):
        s = StringIO()
        with redirect_stdout(s):
            cudf.describe_option("odd_option")
        s.seek(0)
        expected = (
            "odd_option:\n\tAn odd option.\n\t[Default: 1] [Current: 1]\n"
        )
        assert expected == s.read()

    def test_option_description_all(odd_option, even_option):
        s = StringIO()
        with redirect_stdout(s):
            cudf.describe_option()
        s.seek(0)
        expected = (
            "odd_option:\n\tAn odd option.\n\t[Default: 1] [Current: 1]\n"
            "even_option:\n\tAn even option.\n\t[Default: 0] [Current: 0]\n"
        )
        assert expected == s.read()


def test_empty_option_context():
    prev_setting = cudf.get_option("default_integer_bitwidth")
    set_default_integer_bitwidth = random.choice(
        list({32, 64, None} - {prev_setting})
    )
    cudf.set_option("default_integer_bitwidth", set_default_integer_bitwidth)
    with cudf.option_context():
        assert cudf.get_option("default_integer_bitwidth") == 32
        assert cudf.get_option("default_float_bitwidth") is None
        assert cudf.get_option("spill") is os.getenv("CUDF_SPILL", False)
        assert cudf.get_option("copy_on_write") is os.getenv(
            "CUDF_COPY_ON_WRITE", False
        )
        assert cudf.get_option("mode.pandas_compatible") is False
    assert (
        cudf.get_option("default_integer_bitwidth")
        == set_default_integer_bitwidth
    )
    cudf.set_option("default_integer_bitwidth", prev_setting)


def test_option_context():
    prev_pandas_compatible_setting = cudf.get_option("mode.pandas_compatible")
    test_pandas_compatible_value = not prev_pandas_compatible_setting
    cudf.set_option("mode.pandas_compatible", test_pandas_compatible_value)
    assert (
        cudf.get_option("mode.pandas_compatible")
        is test_pandas_compatible_value
    )

    prev_bitwidth_setting = cudf.get_option("default_integer_bitwidth")
    set_default_integer_bitwidth = random.choice(
        list({32, 64, None} - {prev_bitwidth_setting})
    )
    with cudf.option_context(
        "mode.pandas_compatible",
        False,
        "default_integer_bitwidth",
        set_default_integer_bitwidth,
    ):
        assert cudf.get_option("mode.pandas_compatible") is False
        assert (
            cudf.get_option("default_integer_bitwidth")
            == set_default_integer_bitwidth
        )
        cudf.set_option("mode.pandas_compatible", True)
        assert cudf.get_option("mode.pandas_compatible") is True
    assert cudf.get_option("mode.pandas_compatible") is True
    cudf.set_option("mode.pandas_compatible", prev_pandas_compatible_setting)
    assert cudf.get_option("default_integer_bitwidth") == prev_bitwidth_setting


def test_options_context_error():
    with pytest.raises(ValueError):
        with cudf.option_context("mode.pandas_compatible"):
            pass

    with pytest.raises(ValueError):
        with cudf.option_context("mode.pandas_compatible", 1, 2):
            pass
