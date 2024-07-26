# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
An executor for polars logical plans.

This package implements an executor for polars logical plans using
pylibcudf to execute the plans on device.
"""

from __future__ import annotations

import os
import warnings

# We want to avoid initialising the GPU on import. Unfortunately,
# while we still depend on cudf, the default mode is to check things.
# If we set RAPIDS_NO_INITIALIZE, then cudf doesn't do import-time
# validation, good.
# We additionally must set the ptxcompiler environment variable, so
# that we don't check if a numba patch is needed. But if this is done,
# then the patching mechanism warns, and we want to squash that
# warning too.
# TODO: Remove this when we only depend on a pylibcudf package.
os.environ["RAPIDS_NO_INITIALIZE"] = "1"
os.environ["PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED"] = "0"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cudf

    del cudf

from cudf_polars._version import __git_commit__, __version__  # noqa: E402
from cudf_polars.callback import execute_with_cudf  # noqa: E402
from cudf_polars.dsl.translate import translate_ir  # noqa: E402

__all__: list[str] = [
    "execute_with_cudf",
    "translate_ir",
    "__git_commit__",
    "__version__",
]
