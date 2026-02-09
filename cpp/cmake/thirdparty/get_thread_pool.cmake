# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Need to call rapids_cpm_bs_thread_pool to get support for an installed version of thread-pool and
# to support installing it ourselves
function(find_and_configure_thread_pool)
  include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

  # Find or install thread-pool
  rapids_cpm_bs_thread_pool()

endfunction()

find_and_configure_thread_pool()
