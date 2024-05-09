# Copyright (c) 2024, NVIDIA CORPORATION.

import pytest

import dask_cudf


@pytest.mark.parametrize(
    "pkg_data_str", [dask_cudf.__version__, dask_cudf.__git_commit__]
)
def test_version_constants_are_populated(pkg_data_str):
    assert isinstance(pkg_data_str, str)
    assert pkg_data_str == pkg_data_str.strip()
    assert len(pkg_data_str.strip()) > 0
