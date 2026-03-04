# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(params=[True, False])
def as_index(request):
    return request.param


@pytest.fixture(
    params=["min", "max", "idxmin", "idxmax", "count", "sum", "prod", "mean"]
)
def groupby_reduction_methods(request):
    return request.param
