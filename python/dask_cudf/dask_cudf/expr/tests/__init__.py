# Copyright (c) 2024, NVIDIA CORPORATION.

import pytest

from dask import config

from dask_cudf.expr import _expr_support

if not _expr_support or not config.get("dataframe.query-planning", False):
    pytest.skip(allow_module_level=True)
