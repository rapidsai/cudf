# Copyright (c) 2022, NVIDIA CORPORATION.

from contextlib import redirect_stdout
from io import StringIO

import pytest

import cudf


@pytest.fixture(scope="module", autouse=True)
def empty_option_environment():
    old_option_environment = cudf.options._OPTIONS
    cudf.options._OPTIONS = {}
    yield
    cudf.options._OPTIONS = old_option_environment


@pytest.fixture
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


@pytest.fixture
def even_option(empty_option_environment):
    def validator(x):
        if not x % 2 == 0:
            raise ValueError(f"Invalid option value {x}")

    cudf.options._register_option(
        "even_option", 0, "An even option.", validator
    )
    yield
    del cudf.options._OPTIONS["even_option"]


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
    expected = "odd_option:\n\tAn odd option.\n\t[Default: 1] [Current: 1]\n"
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
