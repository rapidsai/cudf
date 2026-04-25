# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds or downloads nvcomp and sets any additional necessary environment variables.
# The CPM build-from-source fallback has been removed since nvcomp cannot be built from source. The
# spacing below is necessary to prevent cmake-format from combining the lint directive with the
# surrounding comments.

# cmake-lint: disable=R0915,R0912,C0103

# This function finds nvcomp and sets any additional necessary environment variables.
function(find_and_configure_nvcomp)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cudf.get_nvcomp")

  # --- 1. Hardcoded version/URL data (replacing versions.json) ---
  set(version "5.2.0.10")

  # Always try proprietary binary (matches prior behavior)
  set(USE_PROPRIETARY_BINARY ON)

  # --- 2. Stage 1: Local search (skip if rapids_cmake_always_download) ---
  include("${rapids-cmake-dir}/find/package.cmake")
  if(NOT rapids_cmake_always_download)
    rapids_find_package(
      nvcomp ${version}
      GLOBAL_TARGETS nvcomp::nvcomp
      BUILD_EXPORT_SET cudf-exports
      INSTALL_EXPORT_SET cudf-exports
      FIND_ARGS QUIET
    )
    if(nvcomp_FOUND)
      message(STATUS "Found nvcomp: ${nvcomp_DIR} (found version ${nvcomp_VERSION})")
    endif()
  endif()
  if(nvcomp_FOUND)
    return()
  endif()

  # --- 3. Stage 2: Proprietary binary download (if not found and USE_PROPRIETARY_BINARY) ---
  include("${rapids-cmake-dir}/cmake/install_lib_dir.cmake")
  rapids_cmake_install_lib_dir(lib_dir)

  set(nvcomp_proprietary_binary OFF)

  if(USE_PROPRIETARY_BINARY AND NOT nvcomp_FOUND)
    # Resolve platform key
    set(platform_key "${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}")
    string(TOLOWER "${platform_key}" platform_key)

    # Look up URL for this platform (hardcoded)
    set(nvcomp_url "")
    if(platform_key STREQUAL "x86_64-linux")
      set(nvcomp_url
          "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-\${version}_cuda\${cuda_version_mapping}-archive.tar.xz"
      )
    elseif(platform_key STREQUAL "aarch64-linux")
      set(nvcomp_url
          "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-sbsa/nvcomp-linux-sbsa-\${version}_cuda\${cuda_version_mapping}-archive.tar.xz"
      )
    else()
      message(
        VERBOSE
        "nvcomp requested usage of a proprietary_binary but none exist for ${CMAKE_SYSTEM_PROCESSOR}"
      )
    endif()

    if(nvcomp_url)
      # Determine CUDA version mapping
      find_package(CUDAToolkit REQUIRED)
      set(cuda_version_mapping "${CUDAToolkit_VERSION_MAJOR}")
      if(CUDAToolkit_VERSION_MAJOR STREQUAL "12")
        set(cuda_version_mapping "12")
      elseif(CUDAToolkit_VERSION_MAJOR STREQUAL "13")
        set(cuda_version_mapping "13")
      endif()

      # Evaluate template
      cmake_language(EVAL CODE "set(nvcomp_url ${nvcomp_url})")

      # Download proprietary binary
      include(FetchContent)
      set(pkg_name "nvcomp_proprietary_binary")
      FetchContent_Declare(${pkg_name} URL ${nvcomp_url})
      FetchContent_MakeAvailable(${pkg_name})
      set(nvcomp_ROOT "${nvcomp_proprietary_binary_SOURCE_DIR}")
      set(nvcomp_proprietary_binary ON)
    endif()

    # Normalize lib64 layout if needed
    if(nvcomp_proprietary_binary)
      if(NOT EXISTS "${nvcomp_ROOT}/${lib_dir}/cmake/nvcomp/nvcomp-config.cmake")
        include(GNUInstallDirs)
        cmake_path(GET lib_dir PARENT_PATH lib_dir_parent)
        cmake_path(GET CMAKE_INSTALL_INCLUDEDIR PARENT_PATH include_dir_parent)
        if(NOT lib_dir_parent STREQUAL include_dir_parent)
          message(
            FATAL_ERROR
              "CMAKE_INSTALL_INCLUDEDIR and CMAKE_INSTALL_LIBDIR must share parent directory"
          )
        endif()

        cmake_path(GET lib_dir FILENAME lib_dir_name)
        set(nvcomp_list_of_target_files
            "nvcomp-targets-common-release.cmake"
            "nvcomp-targets-common.cmake"
            "nvcomp-targets-dynamic-release.cmake"
            "nvcomp-targets-dynamic.cmake"
            "nvcomp-targets-release.cmake"
            "nvcomp-targets-static-release.cmake"
            "nvcomp-targets-static.cmake"
        )
        foreach(filename IN LISTS nvcomp_list_of_target_files)
          if(EXISTS "${nvcomp_ROOT}/lib/cmake/nvcomp/${filename}")
            file(READ "${nvcomp_ROOT}/lib/cmake/nvcomp/${filename}" FILE_CONTENTS)
            string(REPLACE "\$\{_IMPORT_PREFIX\}/lib/" "\$\{_IMPORT_PREFIX\}/${lib_dir_name}/"
                           FILE_CONTENTS ${FILE_CONTENTS}
            )
            file(WRITE "${nvcomp_ROOT}/lib/cmake/nvcomp/${filename}" ${FILE_CONTENTS})
          endif()
        endforeach()
        file(MAKE_DIRECTORY "${nvcomp_ROOT}/${lib_dir_parent}")
        file(RENAME "${nvcomp_ROOT}/lib/" "${nvcomp_ROOT}/${lib_dir}/")
        # Move the `include` dir if necessary as well
        file(RENAME "${nvcomp_ROOT}/include/" "${nvcomp_ROOT}/${CMAKE_INSTALL_INCLUDEDIR}/")
      endif()

      # Record the proprietary root dir location
      set(nvcomp_proprietary_root "${nvcomp_ROOT}")
      cmake_path(NORMAL_PATH nvcomp_proprietary_root)
      set(rapids_cpm_nvcomp_proprietary_root
          "${nvcomp_proprietary_root}"
          CACHE INTERNAL "nvcomp proprietary root dir location"
      )

      # Enforce usage of the local download only
      list(APPEND CMAKE_PREFIX_PATH "${nvcomp_ROOT}/${lib_dir}/cmake/nvcomp")
    endif()
  elseif(DEFINED nvcomp_DIR)
    # Re-configuration: restore state if nvcomp_DIR points to our proprietary root
    cmake_path(NORMAL_PATH nvcomp_DIR)
    if(nvcomp_DIR STREQUAL "${rapids_cpm_nvcomp_proprietary_root}/${lib_dir}/cmake/nvcomp")
      set(nvcomp_proprietary_binary ON)
      set(nvcomp_ROOT "${rapids_cpm_nvcomp_proprietary_root}")
      unset(nvcomp_DIR)
      unset(nvcomp_DIR CACHE)
    endif()
  endif()

  # --- 4. Find proprietary binary + FATAL_ERROR if not found ---
  rapids_find_package(
    nvcomp ${version}
    GLOBAL_TARGETS nvcomp::nvcomp
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
  )

  if(NOT nvcomp_FOUND)
    message(FATAL_ERROR "nvcomp ${version} not found. nvcomp cannot be built from source.")
  endif()

  # --- 5. Target aliases ---
  foreach(name IN ITEMS nvcomp nvcomp_cpu nvcomp_cpu_static nvcomp_static)
    if(NOT TARGET nvcomp::${name} AND TARGET ${name})
      add_library(nvcomp::${name} ALIAS ${name})
    endif()
  endforeach()

  # --- 6. Propagate parent-scope variables ---
  set(nvcomp_VERSION
      ${version}
      PARENT_SCOPE
  )
  set(nvcomp_proprietary_binary
      ${nvcomp_proprietary_binary}
      PARENT_SCOPE
  )

  # --- 7. Install rules (if to_install AND nvcomp_proprietary_binary) ---
  set(to_install ON)
  if(to_install AND nvcomp_proprietary_binary)
    include(GNUInstallDirs)
    install(DIRECTORY "${nvcomp_ROOT}/${lib_dir}/" DESTINATION "${lib_dir}")
    install(DIRECTORY "${nvcomp_ROOT}/${CMAKE_INSTALL_INCLUDEDIR}/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    if(EXISTS "${nvcomp_ROOT}/${CMAKE_INSTALL_BINDIR}")
      install(DIRECTORY "${nvcomp_ROOT}/${CMAKE_INSTALL_BINDIR}/"
              DESTINATION "${CMAKE_INSTALL_BINDIR}"
      )
    endif()
    install(
      FILES "${nvcomp_ROOT}/NOTICE"
      DESTINATION info/
      RENAME NVCOMP_NOTICE
    )
    install(
      FILES "${nvcomp_ROOT}/LICENSE"
      DESTINATION info/
      RENAME NVCOMP_LICENSE
    )
  endif()

  # --- 8. Export tracking ---
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
    BUILD nvcomp "${nvcomp_ROOT}"
    EXPORT_SET cudf-exports
    CONDITION nvcomp_proprietary_binary
  )

endfunction()

find_and_configure_nvcomp()

# Per-thread default stream
if(TARGET nvcomp AND CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(nvcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()
