# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys


def test_cudf_import_no_device(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("CUDA_VISIBLE_DEVICES", "-1")
        output = subprocess.check_call(
            [sys.executable, "-c", "import cudf"],
            cwd="/",
        )
    assert output == 0
