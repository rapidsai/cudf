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

from . import install


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
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass on to the script",
    )

    args = parser.parse_args()

    install()
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
