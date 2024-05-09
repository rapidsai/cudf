# Copyright (c) 2024, NVIDIA CORPORATION.

import custreamz
import pytest


@pytest.mark.parametrize(
    "pkg_data_str", [custreamz.__version__, custreamz.__git_commit__]
)
def test_version_constants_are_populated(pkg_data_str):
    assert isinstance(pkg_data_str, str)
    assert pkg_data_str == pkg_data_str.strip()
    assert len(pkg_data_str.strip()) > 0
