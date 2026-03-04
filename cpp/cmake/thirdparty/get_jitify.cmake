# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Jitify doesn't have a version :/

# This function finds Jitify and sets any additional necessary environment variables.
function(find_and_configure_jitify)
  rapids_cpm_find(
    jitify 2.0.0
    GIT_REPOSITORY https://github.com/NVIDIA/jitify.git
    GIT_TAG 44e978b21fc8bdb6b2d7d8d179523c8350db72e5 # jitify2 branch as of 23rd Aug 2025
    GIT_SHALLOW FALSE
    DOWNLOAD_ONLY TRUE
  )
  set(JITIFY_INCLUDE_DIR
      "${jitify_SOURCE_DIR}"
      PARENT_SCOPE
  )
endfunction()

find_and_configure_jitify()
