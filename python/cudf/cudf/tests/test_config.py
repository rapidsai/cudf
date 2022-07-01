# Copyright (c) 2022, NVIDIA CORPORATION.

import pytest

import cudf


@pytest.fixture(scope="module")
def configuration_demo():
    cudf.register_config(
        "odd_config",
        1,
        "An odd configuration.",
        lambda x: x % 2 == 1,
    )
    yield
    cudf.config._CUDF_CONFIG.pop("odd_config")


@pytest.fixture(scope="module")
def configuration_demo2():
    cudf.register_config(
        "even_config", 0, "An even configuration.", lambda x: x % 2 == 0
    )
    yield
    cudf.config._CUDF_CONFIG.pop("even_config")


def test_config_get_set(configuration_demo):
    assert cudf.get_config("odd_config") == 1
    cudf.config.set_config("odd_config", 101)
    assert cudf.get_config("odd_config") == 101


def test_config_set_invalid(configuration_demo):
    with pytest.raises(ValueError, match="Invalid configuration 0"):
        cudf.set_config("odd_config", 0)


def test_config_description(configuration_demo):
    assert cudf.describe_config("odd_config") == "An odd configuration."


def test_config_description_multi(configuration_demo, configuration_demo2):
    assert cudf.describe_configs() == {
        "odd_config": "An odd configuration.",
        "even_config": "An even configuration.",
    }
