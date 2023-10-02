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

"""
Usage:

python -m xdf.autoload <script.py> <args>
python -m xdf.autoload -m module <args>
"""

import argparse
import runpy
import sys
import tempfile
from contextlib import contextmanager

from ..profiler import Profiler
from . import install

profile_text = """
from xdf.profiler import Profiler
with Profiler() as profiler:
{original_lines}

# Patch the results to shift the line numbers back to the original before the
# profiler injection.
new_results = {{}}

for (lineno, currfile, line), v in profiler._results.items():
    new_results[(lineno - 3, currfile, line)] = v

profiler._results = new_results
profiler.print_stats()
{function_profile_printer}
"""


@contextmanager
def profile(function_profile, line_profile, fn):
    if line_profile:
        with open(fn) as f:
            # Make sure to have consistent spaces instead of tabs
            indented_lines = "".join(
                [(" " * 4) + line.replace("\t", " " * 4) for line in f]
            )

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".py") as f:
            file_contents = profile_text.format(
                original_lines=indented_lines,
                function_profile_printer="profiler.print_per_func_stats()"
                if function_profile
                else "",
            )
            f.write(file_contents.encode())
            f.seek(0)

            yield f.name
    elif function_profile:
        with Profiler() as profiler:
            yield fn
        profiler.print_per_func_stats()
    else:
        yield fn


def main():
    parser = argparse.ArgumentParser(
        prog="python -m xdf.autoload",
        description=(
            "Run a Python script in xdf transparent mode. "
            "In transparent mode, all imports of pandas are "
            "automatically provided by xdf equivalents, using "
            "GPU acceleration with cuDF where possible."
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
        help="Perform per-function profiling of this script.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass on to the script",
    )

    args = parser.parse_args()

    install()
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


if __name__ == "__main__":
    main()
