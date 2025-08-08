# Copyright (c) 2025, NVIDIA CORPORATION.
import pytest


@pytest.fixture(params=[True, False])
def as_index(request):
    return request.param
