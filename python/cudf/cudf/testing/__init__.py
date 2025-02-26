# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import importlib

from cudf.testing.testing import (
    assert_eq,
    assert_frame_equal,
    assert_index_equal,
    assert_neq,
    assert_series_equal,
)


def _import_narwhals_plugin():
    importlib.import_module("cudf.testing.narwhals_tests_plugin")


_import_narwhals_plugin()
