# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
import textwrap


def test_run_cudf_pandas_with_script():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        code = """import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df['a'].sum())"""
        code = textwrap.dedent(
            """
            import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df['a'].sum())
            """
        )
        f.write(code)

        command = f"python -m cudf.pandas {f.name}"
        res = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )

        command = f"python {f.name}"
        expect = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )

    assert res.stdout == expect.stdout


def test_run_cudf_pandas_with_script_with_cmd_args():
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
    assert res.stdout == expect.stdout


def test_cudf_pandas_script_repl():
    def start_repl_process(cmd):
        process = subprocess.Popen(
            cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        return process

    p1 = start_repl_process("python -m cudf.pandas")
    p2 = start_repl_process("python")
    commands = [
        "import pandas as pd\n",
        "print(pd.Series(range(2)).sum())\n",
        "print(pd.Series(range(5)).sum())\n",
    ]

    for c in commands:
        p1.stdin.write(c)
        p2.stdin.write(c)
        p1.stdin.flush()
        p2.stdin.flush()
    res = p1.communicate()[0]
    expect = p1.communicate()[0]
    assert res == expect
    p1.kill()
    p2.kill()
