# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(params=[True, False])
def check_dtype(request):
    """Argument for assert_series_equal, assert_frame_equal"""
    return request.param


@pytest.fixture(params=[True, False])
def check_exact(request):
    """Argument for assert_series_equal, assert_frame_equal"""
    return request.param


@pytest.fixture(params=[True, False])
def check_datetimelike_compat(request):
    """Argument for assert_series_equal, assert_frame_equal"""
    return request.param


@pytest.fixture(params=[True, False])
def check_names(request):
    """Argument for assert_series_equal, assert_frame_equal, assert_index_equal"""
    return request.param
