include(FetchContent)

###################################################################################################
# - cub -------------------------------------------------------------------------------------------

FetchContent_Declare(
    cub
    GIT_REPOSITORY https://github.com/thrust/cub.git
    GIT_TAG        1.9.10
    GIT_SHALLOW    true
    PATCH_COMMAND
        COMMAND    sed -i "494 a histogram_sweep_config.template Init<typename Policy110::HistogramSweepPolicy>()$<SEMICOLON>" cub/device/dispatch/dispatch_histogram.cuh
)

FetchContent_GetProperties(cub)
if(NOT cub_POPULATED)
  FetchContent_Populate(cub)
  # We are not using the cub CMake targets, so no need to call `add_subdirectory()`.
endif()
set(CUB_INCLUDE_DIR "${cub_SOURCE_DIR}" PARENT_SCOPE)

###################################################################################################
# - thrust ----------------------------------------------------------------------------------------

FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    GIT_TAG        1.9.10
    GIT_SHALLOW    true
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
  FetchContent_Populate(thrust)
  # We are not using the thrust CMake targets, so no need to call `add_subdirectory()`.
endif()
set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}" PARENT_SCOPE)
