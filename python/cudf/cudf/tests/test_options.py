# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

import cudf


@pytest.fixture(scope="module")
def odd_option():
    cudf.options._register_option(
        "odd_config",
        1,
        "An odd option.",
        lambda x: x % 2 == 1,
    )
    yield
    cudf.options._CUDF_OPTIONS.pop("odd_config")


@pytest.fixture(scope="module")
def even_option():
    cudf.options._register_option(
        "even_config", 0, "An even option.", lambda x: x % 2 == 0
    )
    yield
    cudf.options._CUDF_OPTIONS.pop("even_config")


def test_config_get_set(odd_option):
    assert cudf.get_option("odd_config") == 1
    cudf.set_option("odd_config", 101)
    assert cudf.get_option("odd_config") == 101


def test_config_set_invalid(odd_option):
    with pytest.raises(ValueError, match="Invalid option 0"):
        cudf.set_option("odd_config", 0)


def test_config_description(odd_option):
    assert cudf.describe_option("odd_config") == "An odd option."


def test_config_description_multi(odd_option, even_option):
    assert cudf.describe_option() == {
        "odd_config": "An odd option.",
        "even_config": "An even option.",
    }
