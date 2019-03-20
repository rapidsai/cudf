# cuDF 0.7.0 (Date TBD)

## New Features

...

## Improvements

...

## Bug Fixes

...


# cuDF 0.6.0 (Date TBD)

## New Features

- PR #760 Raise `FileNotFoundError` instead of `GDF_FILE_ERROR` in `read_csv` if the file does not exist
- PR #539 Add Python bindings for replace function
- PR #823 Add Doxygen configuration to enable building HTML documentation for libcudf C/C++ API
- PR #807 CSV Reader: Add byte_range parameter to specify the range in the input file to be read
- PR #857 Add Tail method for Series/DataFrame and update Head method to use iloc
- PR #858 Add series feature hashing support
- PR #871 CSV Reader: Add support for NA values, including user specified strings
- PR #893 Adds PyArrow based parquet readers / writers to Python, fix category dtype handling, fix arrow ingest buffer size issues
- PR #867 CSV Reader: Add support for ignoring blank lines and comment lines
- PR #887 Add Series digitize method
- PR #895 Add Series groupby
- PR #898 Add DataFrame.groupby(level=0) support
- PR #920 Add feather, JSON, HDF5 readers / writers from PyArrow / Pandas
- PR #888 CSV Reader: Add prefix parameter for column names, used when parsing without a header
- PR #913 Add DLPack support: convert between cuDF DataFrame and DLTensor
- PR #939 Add ORC reader from PyArrow
- PR #918 Add Series.groupby(level=0) support
- PR #906 Add binary and comparison ops to DataFrame
- PR #958 Support unary and binary ops on indexes
- PR #964 Add `rename` method to `DataFrame`, `Series`, and `Index`
- PR #985 Add `Series.to_frame` method
- PR #985 Add `drop=` keyword to reset_index method
- PR #994 Remove references to pygdf
- PR #990 Add external series groupby support
- PR #988 Add top-level merge function to cuDF
- PR #992 Add comparison binaryops to DateTime columns
- PR #996 Replace relative path imports with absolute paths in tests
- PR #995 CSV Reader: Add index_col parameter to specify the column name or index to be used as row labels
- PR #1004 Add `from_gpu_matrix` method to DataFrame
- PR #997 Add property index setter
- PR #1007 Replace relative path imports with absolute paths in cudf
- PR #1013 select columns with df.columns
- PR #1016 Rename Series.unique_count() to nunique() to match pandas API
- PR #947 Prefixsum to handle nulls and float types
- PR #1029 Remove rest of relative path imports
- PR #1021 Add filtered selection with assignment for Dataframes
- PR #872 Adding NVCategory support to cudf apis
- PR #1052 Add left/right_index and left/right_on keywords to merge
- PR #1091 Add `indicator=` and `suffixes=` keywords to merge
- PR #1107 Add unsupported keywords to Series.fillna
- PR #1032 Add string support to cuDF python
- PR #1136 Removed `gdf_concat`
- PR #1153 Added function for getting the padded allocation size for valid bitmask
- PR #1148 Add cudf.sqrt for dataframes and Series
- PR #1159 Add Python bindings for libcudf dlpack functions
- PR #1155 Add __array_ufunc__ for DataFrame and Series for sqrt
- PR #1168 to_frame for series accepts a name argument

## Improvements

- PR #1218 Add dask-cudf page to API docs
- PR #892 Add support for heterogeneous types in binary ops with JIT
- PR #730 Improve performance of `gdf_table` constructor
- PR #561 Add Doxygen style comments to Join CUDA functions
- PR #813 unified libcudf API functions by replacing gpu_ with gdf_
- PR #822 Add support for `__cuda_array_interface__` for ingest
- PR #756 Consolidate common helper functions from unordered map and multimap
- PR #753 Improve performance of groupby sum and average, especially for cases with few groups.
- PR #836 Add ingest support for arrow chunked arrays in Column, Series, DataFrame creation
- PR #763 Format doxygen comments for csv_read_arg struct
- PR #532 CSV Reader: Use type dispatcher instead of switch block
- PR #694 Unit test utilities improvements
- PR #878 Add better indexing to Groupby
- PR #554 Add `empty` method and `is_monotonic` attribute to `Index`
- PR #1040 Fixed up Doxygen comment tags
- PR #909 CSV Reader: Avoid host->device->host copy for header row data
- PR #916 Improved unit testing and error checking for `gdf_column_concat`
- PR #941 Replace `numpy` call in `Series.hash_encode` with `numba`
- PR #942 Added increment/decrement operators for wrapper types
- PR #943 Updated `count_nonzero_mask` to return `num_rows` when the mask is null
- PR #952 Added trait to map C++ type to `gdf_dtype`
- PR #966 Updated RMM submodule.
- PR #998 Add IO reader/writer modules to API docs, fix for missing cudf.Series docs
- PR #1017 concatenate along columns for Series and DataFrames
- PR #1002 Support indexing a dataframe with another boolean dataframe
- PR #1018 Better concatenation for Series and Dataframes
- PR #1036 Use Numpydoc style docstrings
- PR #1047 Adding gdf_dtype_extra_info to gdf_column_view_augmented
- PR #1054 Added default ctor to SerialTrieNode to overcome Thrust issue in CentOS7 + CUDA10
- PR #1024 CSV Reader: Add support for hexadecimal integers in integral-type columns
- PR #1033 Update `fillna()` to use libcudf function `gdf_replace_nulls`
- PR #1066 Added inplace assignment for columns and select_dtypes for dataframes
- PR #1026 CSV Reader: Change the meaning and type of the quoting parameter to match Pandas
- PR #1100 Adds `CUDF_EXPECTS` error-checking macro
- PR #1092 Fix select_dtype docstring
- PR #1111 Added cudf::table
- PR #1108 Sorting for datetime columns
- PR #1120 Return a `Series` (not a `Column`) from `Series.cat.set_categories()`
- PR #1128 CSV Reader: The last data row does not need to be line terminated
- PR #1183 Bump Arrow version to 0.12.1
- PR #1208 Default to CXX11_ABI=ON

## Bug Fixes

- PR #821 Fix flake8 issues revealed by flake8 update
- PR #808 Resolved renamed `d_columns_valids` variable name
- PR #820 SCV Reader: fix the issue where reader adds additional rows when file uses \r\n as a line terminator
- PR #780 CSV Reader: Fix scientific notation parsing and null values for empty quotes
- PR #815 CSV Reader: Fix data parsing when tabs are present in the input CSV file
- PR #850 Fix bug where left joins where the left df has 0 rows causes a crash
- PR #861 Fix memory leak by preserving the boolean mask index
- PR #875 Handle unnamed indexes in to/from arrow functions
- PR #877 Fix ingest of 1 row arrow tables in from arrow function
- PR #876 Added missing `<type_traits>` include
- PR #889 Deleted test_rmm.py which has now moved to RMM repo
- PR #866 Merge v0.5.1 numpy ABI hotfix into 0.6
- PR #917 value_counts return int type on empty columns
- PR #611 Renamed `gdf_reduce_optimal_output_size()` -> `gdf_reduction_get_intermediate_output_size()`
- PR #923 fix index for negative slicing for cudf dataframe and series
- PR #927 CSV Reader: Fix category GDF_CATEGORY hashes not being computed properly
- PR #921 CSV Reader: Fix parsing errors with delim_whitespace, quotations in the header row, unnamed columns
- PR #933 Fix handling objects of all nulls in series creation
- PR #940 CSV Reader: Fix an issue where the last data row is missing when using byte_range
- PR #945 CSV Reader: Fix incorrect datetime64 when milliseconds or space separator are used
- PR #959 Groupby: Problem with column name lookup
- PR #950 Converting dataframe/recarry with non-contiguous arrays
- PR #963 CSV Reader: Fix another issue with missing data rows when using byte_range
- PR #999 Fix 0 sized kernel launches and empty sort_index exception
- PR #993 Fix dtype in selecting 0 rows from objects
- PR #1009 Fix performance regression in `to_pandas` method on DataFrame
- PR #1008 Remove custom dask communication approach
- PR #1001 CSV Reader: Fix a memory access error when reading a large (>2GB) file with date columns
- PR #1019 Binary Ops: Fix error when one input column has null mask but other doesn't
- PR #1014 CSV Reader: Fix false positives in bool value detection
- PR #1034 CSV Reader: Fix parsing floating point precision and leading zero exponents
- PR #1044 CSV Reader: Fix a segfault when byte range aligns with a page
- PR #1058 Added support for `DataFrame.loc[scalar]`
- PR #1060 Fix column creation with all valid nan values
- PR #1073 CSV Reader: Fix an issue where a column name includes the return character
- PR #1090 Updating Doxygen Comments
- PR #1080 Fix dtypes returned from loc / iloc because of lists
- PR #1102 CSV Reader: Minor fixes and memory usage improvements
- PR #1174: Fix release script typo
- PR #1137 Add prebuild script for CI
- PR #1118 Enhanced the `DataFrame.from_records()` feature
- PR #1129 Fix join performance with index parameter from using numpy array
- PR #1145 Issue with .agg call on multi-column dataframes
- PR #908 Some testing code cleanup
- PR #1167 Fix issue with null_count not being set after inplace fillna()
- PR #1184 Fix iloc performance regression
- PR #1185 Support left_on/right_on and also on=str in merge
- PR #1200 Fix allocating bitmasks with numba instead of rmm in allocate_mask function
- PR #1223 gpuCI: Fix label on rapidsai channel on gpu build scripts
- PR #1246 Fix categorical tests that failed due to bad implicit type conversion


# cuDF 0.5.1 (05 Feb 2019)

## Bug Fixes

- PR #842 Avoid using numpy via cimport to prevent ABI issues in Cython compilation


# cuDF 0.5.0 (28 Jan 2019)

## New Features

- PR #722 Add bzip2 decompression support to `read_csv()`
- PR #693 add ZLIB-based GZIP/ZIP support to `read_csv_strings()`
- PR #411 added null support to gdf_order_by (new API) and cudf_table::sort
- PR #525 Added GitHub Issue templates for bugs, documentation, new features, and questions
- PR #501 CSV Reader: Add support for user-specified decimal point and thousands separator to read_csv_strings()
- PR #455 CSV Reader: Add support for user-specified decimal point and thousands separator to read_csv()
- PR #439 add `DataFrame.drop` method similar to pandas
- PR #356 add `DataFrame.transpose` method and `DataFrame.T` property similar to pandas
- PR #505 CSV Reader: Add support for user-specified boolean values
- PR #350 Implemented Series replace function
- PR #490 Added print_env.sh script to gather relevant environment details when reporting cuDF issues
- PR #474 add ZLIB-based GZIP/ZIP support to `read_csv()`
- PR #547 Added melt similar to `pandas.melt()`
- PR #491 Add CI test script to check for updates to CHANGELOG.md in PRs
- PR #550 Add CI test script to check for style issues in PRs
- PR #558 Add CI scripts for cpu-based conda and gpu-based test builds
- PR #524 Add Boolean Indexing
- PR #564 Update python `sort_values` method to use updated libcudf `gdf_order_by` API
- PR #509 CSV Reader: Input CSV file can now be passed in as a text or a binary buffer
- PR #607 Add `__iter__` and iteritems to DataFrame class
- PR #643 added a new api gdf_replace_nulls that allows a user to replace nulls in a column

## Improvements

- PR #426 Removed sort-based groupby and refactored existing groupby APIs. Also improves C++/CUDA compile time.
- PR #461 Add `CUDF_HOME` variable in README.md to replace relative pathing.
- PR #472 RMM: Created centralized rmm::device_vector alias and rmm::exec_policy
- PR #500 Improved the concurrent hash map class to support partitioned (multi-pass) hash table building.
- PR #454 Improve CSV reader docs and examples
- PR #465 Added templated C++ API for RMM to avoid explicit cast to `void**`
- PR #513 `.gitignore` tweaks
- PR #521 Add `assert_eq` function for testing
- PR #502 Simplify Dockerfile for local dev, eliminate old conda/pip envs
- PR #549 Adds `-rdynamic` compiler flag to nvcc for Debug builds
- PR #472 RMM: Created centralized rmm::device_vector alias and rmm::exec_policy
- PR #577 Added external C++ API for scatter/gather functions
- PR #500 Improved the concurrent hash map class to support partitioned (multi-pass) hash table building
- PR #583 Updated `gdf_size_type` to `int`
- PR #500 Improved the concurrent hash map class to support partitioned (multi-pass) hash table building
- PR #617 Added .dockerignore file. Prevents adding stale cmake cache files to the docker container
- PR #658 Reduced `JOIN_TEST` time by isolating overflow test of hash table size computation
- PR #664 Added Debuging instructions to README
- PR #651 Remove noqa marks in `__init__.py` files
- PR #671 CSV Reader: uncompressed buffer input can be parsed without explicitly specifying compression as None
- PR #684 Make RMM a submodule
- PR #718 Ensure sum, product, min, max methods pandas compatibility on empty datasets
- PR #720 Refactored Index classes to make them more Pandas-like, added CategoricalIndex
- PR #749 Improve to_arrow and from_arrow Pandas compatibility
- PR #766 Remove TravisCI references, remove unused variables from CMake, fix ARROW_VERSION in Cmake
- PR #773 Add build-args back to Dockerfile and handle dependencies based on environment yml file
- PR #781 Move thirdparty submodules to root and symlink in /cpp
- PR #843 Fix broken cudf/python API examples, add new methods to the API index

## Bug Fixes

- PR #569 CSV Reader: Fix days being off-by-one when parsing some dates
- PR #531 CSV Reader: Fix incorrect parsing of quoted numbers
- PR #465 Added templated C++ API for RMM to avoid explicit cast to `void**`
- PR #473 Added missing <random> include
- PR #478 CSV Reader: Add api support for auto column detection, header, mangle_dupe_cols, usecols
- PR #495 Updated README to correct where cffi pytest should be executed
- PR #501 Fix the intermittent segfault caused by the `thousands` and `compression` parameters in the csv reader
- PR #502 Simplify Dockerfile for local dev, eliminate old conda/pip envs
- PR #512 fix bug for `on` parameter in `DataFrame.merge` to allow for None or single column name
- PR #511 Updated python/cudf/bindings/join.pyx to fix cudf merge printing out dtypes
- PR #513 `.gitignore` tweaks
- PR #521 Add `assert_eq` function for testing
- PR #537 Fix CMAKE_CUDA_STANDARD_REQURIED typo in CMakeLists.txt
- PR #447 Fix silent failure in initializing DataFrame from generator
- PR #545 Temporarily disable csv reader thousands test to prevent segfault (test re-enabled in PR #501)
- PR #559 Fix Assertion error while using `applymap` to change the output dtype
- PR #575 Update `print_env.sh` script to better handle missing commands
- PR #612 Prevent an exception from occuring with true division on integer series.
- PR #630 Fix deprecation warning for `pd.core.common.is_categorical_dtype`
- PR #622 Fix Series.append() behaviour when appending values with different numeric dtype
- PR #603 Fix error while creating an empty column using None.
- PR #673 Fix array of strings not being caught in from_pandas
- PR #644 Fix return type and column support of dataframe.quantile()
- PR #634 Fix create `DataFrame.from_pandas()` with numeric column names
- PR #654 Add resolution check for GDF_TIMESTAMP in Join
- PR #648 Enforce one-to-one copy required when using `numba>=0.42.0`
- PR #645 Fix cmake build type handling not setting debug options when CMAKE_BUILD_TYPE=="Debug"
- PR #669 Fix GIL deadlock when launching multiple python threads that make Cython calls
- PR #665 Reworked the hash map to add a way to report the destination partition for a key
- PR #670 CMAKE: Fix env include path taking precedence over libcudf source headers
- PR #674 Check for gdf supported column types
- PR #677 Fix 'gdf_csv_test_Dates' gtest failure due to missing nrows parameter
- PR #604 Fix the parsing errors while reading a csv file using `sep` instead of `delimiter`.
- PR #686 Fix converting nulls to NaT values when converting Series to Pandas/Numpy
- PR #689 CSV Reader: Fix behavior with skiprows+header to match pandas implementation
- PR #691 Fixes Join on empty input DFs
- PR #706 CSV Reader: Fix broken dtype inference when whitespace is in data
- PR #717 CSV reader: fix behavior when parsing a csv file with no data rows
- PR #724 CSV Reader: fix build issue due to parameter type mismatch in a std::max call
- PR #734 Prevents reading undefined memory in gpu_expand_mask_bits numba kernel
- PR #747 CSV Reader: fix an issue where CUDA allocations fail with some large input files
- PR #750 Fix race condition for handling NVStrings in CMake
- PR #719 Fix merge column ordering
- PR #770 Fix issue where RMM submodule pointed to wrong branch and pin other to correct branches
- PR #778 Fix hard coded ABI off setting
- PR #784 Update RMM submodule commit-ish and pip paths
- PR #794 Update `rmm::exec_policy` usage to fix segmentation faults when used as temprory allocator.
- PR #800 Point git submodules to branches of forks instead of exact commits


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

