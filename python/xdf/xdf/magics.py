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


from IPython.core.magic import Magics, cell_magic, magics_class

from .profiler import Profiler

profile_patch_text = """
# Patch the results to shift the line numbers back to the original before the
# profiler injection.
new_results = {}

for (lineno, currfile, line), v in profiler._results.items():
    new_results[(lineno - 2, currfile, line)] = v

profiler._results = new_results
profiler.print_stats()
"""


@magics_class
class XDFMagics(Magics):
    @cell_magic
    def xdf_profile(self, _, cell):
        with Profiler() as profiler:
            get_ipython().run_cell(cell)  # noqa: F821
        profiler.print_per_func_stats()

    @cell_magic
    def xdf_line_profile(self, _, cell):
        cell_split = cell.split("\n")
        cell_split = [
            " " * 4 + line.replace("\t", " " * 4) for line in cell_split
        ]
        cell_split.insert(0, "from xdf.profiler import Profiler")
        cell_split.insert(1, "with Profiler() as profiler:")
        new_cell = "\n".join(cell_split) + "\n" + profile_patch_text
        get_ipython().run_cell(new_cell)  # noqa: F821


def load_ipython_extension(ip):
    import xdf.autoload

    xdf.autoload.install()
    ip.register_magics(XDFMagics)
