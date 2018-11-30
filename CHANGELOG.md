# cuDF 0.4.0 (DATE TBD)

## New Features

- PR #398 add pandas-compatible `DataFrame.shape()` and `Series.shape()`
- PR #394 New documentation feature "10 Minutes to cuDF"
- PR #361 CSV Reader: Add support to read strings with delimiters

## Improvements

 - PR #266 use faster CUDA-accelerated DataFrame column/Series concatenation.
 - PR #379 new C++ `type_dispatcher` reduces code complexity in supporting many data types.
 - PR #349 Improve performance for creating columns from memoryview objects
 
 
## Bug Fixes

 - PR #396 added missing output buffer in validity tests for groupbys.
 - PR #408 Docker file updates for source reorganization
 - PR #417 Fix `map_test` failure with CUDA 10
 - PR #414 Fix CMake installation include file paths
 - PR #418 Properly cast string dtypes to programmatic dtypes when instantiating columns
 - PR #427 Fix and tests for Concatenation illegal memory access with nulls
 

# cuDF 0.3.0 (23 Nov 2018)

## New Features

## Improvements

 - PR #354 source code refactored for better organization. CMake build system overhaul.

## Bug Fixes

# cuDF 0.2.0 (???)

## New Features

## Improvements

## Bug Fixes

# cuDF 0.1.0 (???)

## New Features

## Improvements

## Bug Fixes
