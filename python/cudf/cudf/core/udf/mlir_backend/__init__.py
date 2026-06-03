# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""MLIR / numba_cuda_mlir UDF backend (scaffolding).

Future PRs in this series will populate this package with typing/lowering
modules that register `cudf` UDF types (``MaskedType``, ``StringView``,
``GroupType``, etc.) with ``numba_cuda_mlir``. To avoid circular imports
through ``cudf.core.udf.utils``, registration is performed by the parent
``cudf.core.udf`` ``__init__`` once the modules exist; this file is
intentionally empty.
"""
