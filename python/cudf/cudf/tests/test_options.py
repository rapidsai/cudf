# Copyright (c) 2022-2023, NVIDIA CORPORATION.

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


@pytest.mark.parametrize("default_integer_bitwidth", [32, 64, None])
def test_empty_option_context(default_integer_bitwidth):
    prev_setting = cudf.get_option("default_integer_bitwidth")
    cudf.set_option("default_integer_bitwidth", default_integer_bitwidth)
    with cudf.option_context():
        assert (
            cudf.get_option("default_integer_bitwidth")
            == default_integer_bitwidth
        )

    assert (
        cudf.get_option("default_integer_bitwidth") == default_integer_bitwidth
    )
    cudf.set_option("default_integer_bitwidth", prev_setting)


@pytest.mark.parametrize("pandas_compatible", [True, False])
@pytest.mark.parametrize("default_integer_bitwidth", [32, 64])
def test_option_context(pandas_compatible, default_integer_bitwidth):
    prev_pandas_compatible_setting = cudf.get_option("mode.pandas_compatible")
    prev_width_setting = cudf.get_option("default_integer_bitwidth")

    with cudf.option_context(
        "mode.pandas_compatible",
        pandas_compatible,
        "default_integer_bitwidth",
        default_integer_bitwidth,
    ):
        assert cudf.get_option("mode.pandas_compatible") is pandas_compatible
        assert (
            cudf.get_option("default_integer_bitwidth")
            is default_integer_bitwidth
        )

    assert (
        cudf.get_option("mode.pandas_compatible")
        is prev_pandas_compatible_setting
    )
    assert cudf.get_option("default_integer_bitwidth") is prev_width_setting


def test_options_context_error():
    with pytest.raises(ValueError):
        with cudf.option_context("mode.pandas_compatible"):
            pass

    with pytest.raises(ValueError):
        with cudf.option_context("mode.pandas_compatible", 1, 2):
            pass
