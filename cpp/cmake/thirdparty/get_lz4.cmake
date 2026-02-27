# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone lz4
function(find_and_configure_lz4)

  set(CPM_DOWNLOAD_lz4 ON)
  rapids_cpm_find(
    lz4 dev
    GLOBAL_TARGETS lz4
    CPM_ARGS
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG 446a35f
    GIT_SHALLOW TRUE
  )

  if(lz4_ADDED)
    add_library(
      lz4_objects OBJECT
      ${lz4_SOURCE_DIR}/lib/lz4.c ${lz4_SOURCE_DIR}/lib/lz4file.c ${lz4_SOURCE_DIR}/lib/lz4frame.c
      ${lz4_SOURCE_DIR}/lib/lz4hc.c ${lz4_SOURCE_DIR}/lib/xxhash.c
    )
    target_include_directories(lz4_objects PUBLIC $<BUILD_INTERFACE:${lz4_SOURCE_DIR}/lib>)
  endif()

endfunction()

find_and_configure_lz4()
