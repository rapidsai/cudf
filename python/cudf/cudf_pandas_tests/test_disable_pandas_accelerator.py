# Copyright (c) 2025, NVIDIA CORPORATION.

import os
import subprocess

from cudf.testing import _utils as utils


def test_disable_pandas_accelerator_multi_threaded():
    data_directory = os.path.dirname(os.path.abspath(__file__))
    # Create a copy of the current environment variables
    env = os.environ.copy()

    with utils.cudf_timeout(20):
        sp_completed = subprocess.run(
            [
                "python",
                "-m",
                "cudf.pandas",
                data_directory + "/data/disable_cudf_pandas_multi_thread.py",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
    assert sp_completed.returncode == 0
    output = sp_completed.stdout

    for string in [
        "Result from thread 1: 0<class 'type'>",
        "Result from thread 2: 1<class 'type'>",
        "Result from thread 3: 2<class 'type'>",
        "Result from thread 4: 3<class 'type'>",
    ]:
        assert string in output
