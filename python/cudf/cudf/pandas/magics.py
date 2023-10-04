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


try:
    from IPython.core.magic import Magics, cell_magic, magics_class

    from .profiler import Profiler, lines_with_profiling


    @magics_class
    class CudfPandasMagics(Magics):
        @cell_magic("cudf.pandas.profile")
        def profile(self, _, cell):
            with Profiler() as profiler:
                get_ipython().run_cell(cell)  # noqa: F821
            profiler.print_per_func_stats()

        @cell_magic("cudf.pandas.line_profile")
        def line_profile(self, _, cell):
            new_cell = lines_with_profiling(cell.split("\n"))
            get_ipython().run_cell(new_cell)  # noqa: F821


    def load_ipython_extension(ip):
        from . import install

        install()
        ip.register_magics(CudfPandasMagics)

except ImportError:

    def load_ipython_extension(ip):
        pass
