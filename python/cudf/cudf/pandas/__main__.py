# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Usage:

python -m cudf.pandas <script.py> <args>
python -m cudf.pandas -m module <args>
"""

import argparse
import runpy
import sys
import tempfile
from contextlib import contextmanager

from . import install
from .profiler import Profiler, lines_with_profiling


@contextmanager
def profile(function_profile, line_profile, fn):
    if line_profile:
        with open(fn) as f:
            lines = f.readlines()

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".py") as f:
            f.write(lines_with_profiling(lines, function_profile).encode())
            f.seek(0)

            yield f.name
    elif function_profile:
        with Profiler() as profiler:
            yield fn
        profiler.print_per_function_stats()
    else:
        yield fn


def main():
    parser = argparse.ArgumentParser(
        prog="python -m cudf.pandas",
        description=(
            "Run a Python script with Pandas Accelerator Mode enabled. "
            "In Pandas Accelerator Mode, all imports of pandas will "
            "automatically use GPU accelerated cuDF equivalents where "
            "possible."
        ),
    )

    parser.add_argument(
        "-m",
        dest="module",
        nargs=1,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Perform per-function profiling of this script.",
    )
    parser.add_argument(
        "--line-profile",
        action="store_true",
        help="Perform per-line profiling of this script.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass on to the script",
    )

    args = parser.parse_args()

    rmm_mode = install()
    with profile(args.profile, args.line_profile, args.args[0]) as fn:
        args.args[0] = fn
        if args.module:
            (module,) = args.module
            # run the module passing the remaining arguments
            # as if it were run with python -m <module> <args>
            sys.argv[:] = [module] + args.args  # not thread safe?
            runpy.run_module(module, run_name="__main__")
        elif len(args.args) >= 1:
            # Remove ourself from argv and continue
            sys.argv[:] = args.args
            runpy.run_path(args.args[0], run_name="__main__")

    if "managed" in rmm_mode:
        for key in {
            "column_view::get_data",
            "mutable_column_view::get_data",
            "gather",
            "hash_join",
        }:
            from cudf._lib import pylibcudf

            pylibcudf.experimental.enable_prefetching(key)


if __name__ == "__main__":
    main()
