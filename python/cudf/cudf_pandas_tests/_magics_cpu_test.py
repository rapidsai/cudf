# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def ipython_magics_cpu_test():
    import warnings

    from IPython.core.interactiveshell import InteractiveShell

    # Use in-memory history file to avoid file handles leaking
    # https://github.com/pandas-dev/pandas/pull/35711
    from traitlets.config import Config  # isort:skip

    warnings.filterwarnings("ignore", category=UserWarning)

    c = Config()
    c.HistoryManager.hist_file = ":memory:"

    ip = InteractiveShell(config=c)
    ip.run_line_magic("load_ext", "cudf.pandas")

    # confirm pandas is not aliased
    ip.run_cell("import pandas as pd; s = pd.Series(range(5))")
    result = ip.run_cell("assert not hasattr(s, '_fsproxy_state')")
    result.raise_error()


if __name__ == "__main__":
    ipython_magics_cpu_test()
