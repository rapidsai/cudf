# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import subprocess
import sys

import pytest


# Proxy for whether we should expect to see XDF objects
# without trying to import cuDF, (which could poison the global environment)
def _gpu_available():
    try:
        import rmm

        return rmm._cuda.gpu.getDeviceCount() >= 1
    except ImportError:
        return False


LOCATION = pathlib.Path(__file__).absolute().parent


@pytest.mark.skipif(
    not _gpu_available(), reason="Skipping test if a GPU isn't available."
)
def test_magics_gpu():
    sp_completed = subprocess.run(
        [sys.executable, LOCATION / "_magics_gpu_test.py"], capture_output=True
    )
    assert sp_completed.stderr.decode() == ""


@pytest.mark.skip(
    "This test was viable when xdf was separate from cudf, but now that it is "
    "a subpackage we always require a GPU to be present and cannot run this "
    "test."
)
def test_magics_cpu():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    sp_completed = subprocess.run(
        [sys.executable, LOCATION / "_magics_cpu_test.py"],
        capture_output=True,
        env=env,
    )
    assert sp_completed.stderr.decode() == ""
