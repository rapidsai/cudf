
# cuDF 0.5.0 (Date TBD)

## New Features

- PR #411 added null support to gdf_order_by (new API) and cudf_table::sort
- PR #455 CSV Reader: Add support for user-specified decimal point and thousands separator
- PR #439 add `DataFrame.drop` method similar to pandas

## Improvements

- PR #472 RMM: Created centralized rmm::device_vector alias and rmm::exec_policy
- PR #426 Removed sort-based groupby and refactored existing groupby APIs. Also improves C++/CUDA compile time.

## Bug Fixes


# cuDF 0.4.0 (05 Dec 2018)

## New Features

- PR #398 add pandas-compatible `DataFrame.shape()` and `Series.shape()`
- PR #394 New documentation feature "10 Minutes to cuDF"
- PR #361 CSV Reader: Add support for strings with delimiters

## Improvements

 - PR #436 Improvements for type_dispatcher and wrapper structs
 - PR #429 Add CHANGELOG.md (this file)
 - PR #266 use faster CUDA-accelerated DataFrame column/Series concatenation.
 - PR #379 new C++ `type_dispatcher` reduces code complexity in supporting many data types.
 - PR #349 Improve performance for creating columns from memoryview objects
 - PR #445 Update reductions to use type_dispatcher. Adds integer types support to sum_of_squares. 
 - PR #448 Improve installation instructions in README.md
 - PR #456 Change default CMake build to Release, and added option for disabling compilation of tests
 
## Bug Fixes

 - PR #444 Fix csv_test CUDA too many resources requested fail. 
 - PR #396 added missing output buffer in validity tests for groupbys.
 - PR #408 Dockerfile updates for source reorganization
 - PR #437 Add cffi to Dockerfile conda env, fixes "cannot import name 'librmm'"
 - PR #417 Fix `map_test` failure with CUDA 10
 - PR #414 Fix CMake installation include file paths
 - PR #418 Properly cast string dtypes to programmatic dtypes when instantiating columns
 - PR #427 Fix and tests for Concatenation illegal memory access with nulls
 

# cuDF 0.3.0 (23 Nov 2018)

## New Features

 - PR #336 CSV Reader string support

## Improvements
 
 - PR #354 source code refactored for better organization. CMake build system overhaul. Beginning of transition to Cython bindings.
 - PR #290 Add support for typecasting to/from datetime dtype
 - PR #323 Add handling pyarrow boolean arrays in input/out, add tests
 - PR #325 GDF_VALIDITY_UNSUPPORTED now returned for algorithms that don't support non-empty valid bitmasks
 - PR #381 Faster InputTooLarge Join test completes in ms rather than minutes. 
 - PR #373 .gitignore improvements
 - PR #367 Doc cleanup & examples for DataFrame methods
 - PR #333 Add Rapids Memory Manager documentation
 - PR #321 Rapids Memory Manager adds file/line location logging and convenience macros
 - PR #334 Implement DataFrame `__copy__` and `__deepcopy__`
 - PR #271 Add NVTX ranges to pygdf 
 - PR #311 Document system requirements for conda install

## Bug Fixes

 - PR #337 Retain index on `scale()` function
 - PR #344 Fix test failure due to PyArrow 0.11 Boolean handling
 - PR #364 Remove noexcept from managed_allocator;  CMakeLists fix for NVstrings
 - PR #357 Fix bug that made all series be considered booleans for indexing
 - PR #351 replace conda env configuration for developers
 - PRs #346 #360 Fix CSV reading of negative numbers
 - PR #342 Fix CMake to use conda-installed nvstrings
 - PR #341 Preserve categorical dtype after groupby aggregations 
 - PR #315 ReadTheDocs build update to fix missing libcuda.so
 - PR #320 FIX out-of-bounds access error in reductions.cu 
 - PR #319 Fix out-of-bounds memory access in libcudf count_valid_bits
 - PR #303 Fix printing empty dataframe

# cuDF 0.2.0 and cuDF 0.1.0

These were initial releases of cuDF based on previously separate pyGDF and libGDF libraries.

