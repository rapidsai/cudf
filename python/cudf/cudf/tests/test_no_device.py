# Copyright (c) 2024, NVIDIA CORPORATION.
import os
import subprocess


def test_cudf_import_no_device():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    output = subprocess.run(
        ["python", "-c", "import cudf"],
        env=env,
        capture_output=True,
        text=True,
        cwd="/",
    )
    assert output.returncode == 0
