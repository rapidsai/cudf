# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# This function finds rtcx
function(find_and_configure_rtcx VERSION)

  rapids_cpm_find(
    rtcx ${VERSION}
    GLOBAL_TARGETS rtcx::rtcx
    CPM_ARGS
    GIT_REPOSITORY https://github.com/vyasr/librtcx.git
    GIT_TAG 31a72956cca60fa6039e6af746ed4e9771cca740
    GIT_SHALLOW FALSE
    EXCLUDE_FROM_ALL ${CUDF_EXCLUDE_DEPS_FROM_ALL}
  )

  # When CPM fetches from source (add_subdirectory), embed.cmake is not auto-included. Include it
  # explicitly so add_embed/embed_includes/embed functions are available.
  if(rtcx_ADDED OR DEFINED CPM_rtcx_SOURCE)
    include("${rtcx_SOURCE_DIR}/embed.cmake")
  endif()

endfunction()

set(RTCX_MIN_VERSION_cudf "0.1")
find_and_configure_rtcx(${RTCX_MIN_VERSION_cudf})
