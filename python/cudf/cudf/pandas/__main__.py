# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Usage:

python -m cudf.pandas <script.py> <args>
python -m cudf.pandas -m module <args>
"""

import argparse
import code
import runpy
import sys
import tempfile
from contextlib import contextmanager

from . import install
from .profiler import Profiler, lines_with_profiling


@contextmanager
def profile(function_profile, line_profile, fn):
    if fn is None and (line_profile or function_profile):
        raise RuntimeError("Enabling the profiler requires a script name.")
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
        "-c",
        dest="cmd",
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

    if args.cmd:
        f = tempfile.NamedTemporaryFile(mode="w+b", suffix=".py")
        f.write(args.cmd[0].encode())
        f.seek(0)
        args.args.insert(0, f.name)

    install()

    script_name = args.args[0] if len(args.args) > 0 else None
    with profile(args.profile, args.line_profile, script_name) as fn:
        if script_name is not None:
            args.args[0] = fn
        if args.module:
            (module,) = args.module
            # run the module passing the remaining arguments
            # as if it were run with python -m <module> <args>
            sys.argv[:] = [module, *args.args]  # not thread safe?
            runpy.run_module(module, run_name="__main__")
        elif len(args.args) >= 1:
            # Remove ourself from argv and continue
            sys.argv[:] = args.args
            runpy.run_path(args.args[0], run_name="__main__")
        else:
            if sys.stdin.isatty():
                banner = f"Python {sys.version} on {sys.platform}"
                site_import = not sys.flags.no_site
                if site_import:
                    cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
                    banner += "\n" + cprt
            else:
                # Don't show prompts or banners if stdin is not a TTY
                sys.ps1 = ""
                sys.ps2 = ""
                banner = ""

            # Launch an interactive interpreter
            code.interact(banner=banner, exitmsg="")


if __name__ == "__main__":
    main()
