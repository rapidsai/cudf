# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile


def test_run_cudf_pandas_with_cli():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        code = """import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df['a'].sum())"""
        f.write(code)
        f.seek(0)

        command = f"python -m cudf.pandas {f.name}"
        res = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )

        f.seek(0)
        command = f"python {f.name}"
        expect = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )

        assert res.stdout == expect.stdout


def test_run_cudf_pandas_with_cli_with_cmd_args():
    input_args_and_code = '''-c "import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df['a'].sum())"'''
    res = subprocess.run(
        "python -m cudf.pandas " + input_args_and_code,
        shell=True,
        capture_output=True,
        text=True,
    )
    expect = subprocess.run(
        "python " + input_args_and_code,
        shell=True,
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    assert res.stdout == expect.stdout
