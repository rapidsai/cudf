# cuDF 0.12.0 (Date TBD)

## New Features

- PR #3284 Add gpu-accelerated parquet writer
- PR #3336 Add `from_dlpack` and `to_dlpack`
- PR #3555 Add column names support to libcudf++ io readers and writers

## Improvements

- PR #3502 ORC reader: add option to read DECIMALs as INT64
- PR #3461 Add a new overload to allocate_like() that takes explicit type and size params.
- PR #3590 Specialize hash functions for floating point
- PR #3569 Use `np.asarray` in `StringColumn.deserialize`
- PR #3553 Support Python NoneType in numeric binops

- PR #3431 Port NVStrings translate to cudf strings column

## Bug Fixes

- PR #3550 Update Java package to 0.12
- PR #3549 Fix index name issue with iloc with RangeIndex
- PR #3562 Fix 4GB limit for gzipped-compressed csv files
- PR #3563 Use `__cuda_array_interface__` for serialization
- PR #3548 Replaced CUDA_RT_CALL with CUDA_TRY


# cuDF 0.11.0 (11 Dec 2019)

## New Features

- PR #2905 Added `Series.median()` and null support for `Series.quantile()`
- PR #2930 JSON Reader: Support ARROW_RANDOM_FILE input
- PR #2956 Add `cudf::stack` and `cudf::tile`
- PR #2980 Added nvtext is_vowel/is_consonant functions
- PR #2987 Add `inplace` arg to `DataFrame.reset_index` and `Series`
- PR #3011 Added libcudf++ transition guide
- PR #3129 Add strings column factory from `std::vector`s
- PR #3054 Add parquet reader support for decimal data types
- PR #3022 adds DataFrame.astype for cuDF dataframes
- PR #2962 Add isnull(), notnull() and related functions
- PR #3025 Move search files to legacy
- PR #3068 Add `scalar` class
- PR #3094 Adding `any` and `all` support from libcudf
- PR #3130 Define and implement new `column_wrapper`
- PR #3143 Define and implement new copying APIs `slice` and `split`
- PR #3161 Move merge files to legacy
- PR #3079 Added support to write ORC files given a local path
- PR #3192 Add dtype param to cast `DataFrame` on init
- PR #3213 Port cuIO to libcudf++
- PR #3222 Add nvtext character tokenizer
- PR #3223 Java expose underlying buffers
- PR #3300 Add `DataFrame.insert`
- PR #3263 Define and implement new `valid_if`
- PR #3278 Add `to_host` utility to copy `column_view` to host
- PR #3087 Add new cudf::experimental bool8 wrapper
- PR #3219 Construct column from column_view
- PR #3250 Define and implement new merge APIs
- PR #3144 Define and implement new hashing APIs `hash` and `hash_partition`
- PR #3229 Define and implement new search APIs
- PR #3308 java add API for memory usage callbacks
- PR #2691 Row-wise reduction and scan operations via CuPy
- PR #3291 Add normalize_nans_and_zeros
- PR #3187 Define and implement new replace APIs
- PR #3356 Add vertical concatenation for table/columns
- PR #3344 java split API
- PR #2791 Add `groupby.std()`
- PR #3368 Enable dropna argument in dask_cudf groupby
- PR #3298 add null replacement iterator for column_device_view
- PR #3297 Define and implement new groupby API.
- PR #3396 Update device_atomics with new bool8 and timestamp specializations
- PR #3411 Java host memory management API
- PR #3393 Implement df.cov and enable covariance/correlation in dask_cudf
- PR #3401 Add dask_cudf ORC writer (to_orc)
- PR #3331 Add copy_if_else
- PR #3427 Define and Implement new multi-search API
- PR #3442 Add Bool-index + Multi column + DataFrame support for set-item
- PR #3172 Define and implement new fill/repeat/copy_range APIs
- PR #3497 Add DataFrame.drop(..., inplace=False) argument
- PR #3469 Add string functionality for replace API

## Improvements

- PR #2904 Move gpu decompressors to cudf::io namespace
- PR #2977 Moved old C++ test utilities to legacy directory.
- PR #2965 Fix slow orc reader perf with large uncompressed blocks
- PR #2995 Move JIT type utilities to legacy directory
- PR #2927 Add ``Table`` and ``TableView`` extension classes that wrap legacy cudf::table
- PR #3005 Renames `cudf::exp` namespace to `cudf::experimental`
- PR #3008 Make safe versions of `is_null` and `is_valid` in `column_device_view`
- PR #3026 Move fill and repeat files to legacy
- PR #3027 Move copying.hpp and related source to legacy folder
- PR #3014 Snappy decompression optimizations
- PR #3032 Use `asarray` to coerce indices to a NumPy array
- PR #2996 IO Readers: Replace `cuio::device_buffer` with `rmm::device_buffer`
- PR #3051 Specialized hash function for strings column
- PR #3065 Select and Concat for cudf::experimental::table
- PR #3080 Move `valid_if.cuh` to `legacy/`
- PR #3052 Moved replace.hpp functionality to legacy
- PR #3091 Move join files to legacy
- PR #3092 Implicitly init RMM if Java allocates before init
- PR #3029 Update gdf_ numeric types with stdint and move to cudf namespace
- PR #3052 Moved replace.hpp functionality to legacy
- PR #2955 Add cmake option to only build for present GPU architecture
- PR #3070 Move functions.h and related source to legacy
- PR #2951 Allow set_index to handle a list of column names
- PR #3093 Move groupby files to legacy
- PR #2988 Removing GIS functionality (now part of cuSpatial library)
- PR #3067 Java method to return size of device memory buffer
- PR #3083 Improved some binary operation tests to include null testing.
- PR #3084 Update to arrow-cpp and pyarrow 0.15.0
- PR #3071 Move cuIO to legacy
- PR #3126 Round 2 of snappy decompression optimizations
- PR #3046 Define and implement new copying APIs `empty_like` and `allocate_like`
- PR #3128 Support MultiIndex in DataFrame.join
- PR #2971 Added initial gather and scatter methods for strings_column_view
- PR #3133 Port NVStrings to cudf column: count_characters and count_bytes
- PR #2991 Added strings column functions concatenate and join_strings
- PR #3028 Define and implement new `gather` APIs.
- PR #3135 Add nvtx utilities to cudf::nvtx namespace
- PR #3021 Java host side concat of serialized buffers
- PR #3138 Move unary files to legacy
- PR #3170 Port NVStrings substring functions to cudf strings column
- PR #3159 Port NVStrings is-chars-types function to cudf strings column
- PR #3154 Make `table_view_base.column()` const and add `mutable_table_view.column()`
- PR #3175 Set cmake cuda version variables
- PR #3171 Move deprecated error macros to legacy
- PR #3191 Port NVStrings integer convert ops to cudf column
- PR #3189 Port NVStrings find ops to cudf column
- PR #3352 Port NVStrings convert float functions to cudf strings column
- PR #3193 Add cuPy as a formal dependency
- PR #3195 Support for zero columned `table_view`
- PR #3165 Java device memory size for string category
- PR #3205 Move transform files to legacy
- PR #3202 Rename and move error.hpp to public headers
- PR #2878 Use upstream merge code in dask_cudf
- PR #3217 Port NVStrings upper and lower case conversion functions
- PR #3350 Port NVStrings booleans convert functions
- PR #3231 Add `column::release()` to give up ownership of contents.
- PR #3157 Use enum class rather than enum for mask_allocation_policy
- PR #3232 Port NVStrings datetime conversion to cudf strings column
- PR #3136 Define and implement new transpose API
- PR #3237 Define and implement new transform APIs
- PR #3245 Move binaryop files to legacy
- PR #3241 Move stream_compaction files to legacy
- PR #3166 Move reductions to legacy
- PR #3261 Small cleanup: remove `== true`
- PR #3271 Update rmm API based on `rmm.reinitialize(...)` change
- PR #3266 Remove optional checks for CuPy
- PR #3268 Adding null ordering per column feature when sorting
- PR #3239 Adding floating point specialization to comparators for NaNs
- PR #3270 Move predicates files to legacy
- PR #3281 Add to_host specialization for strings in column test utilities
- PR #3282 Add `num_bitmask_words`
- PR #3252 Add new factory methods to include passing an existing null mask
- PR #3288 Make `bit.cuh` utilities usable from host code.
- PR #3287 Move rolling windows files to legacy
- PR #3182 Define and implement new unary APIs `is_null` and `is_not_null`
- PR #3314 Drop `cython` from run requirements
- PR #3301 Add tests for empty column wrapper.
- PR #3294 Update to arrow-cpp and pyarrow 0.15.1
- PR #3310 Add `row_hasher` and `element_hasher` utilities
- PR #3272 Support non-default streams when creating/destroying hash maps
- PR #3286 Clean up the starter code on README
- PR #3332 Port NVStrings replace to cudf strings column
- PR #3354 Define and implement new `scatter` APIs
- PR #3322 Port NVStrings pad operations to cudf strings column
- PR #3345 Add cache member for number of characters in string_view class
- PR #3299 Define and implement new `is_sorted` APIs
- PR #3328 Partition by stripes in dask_cudf ORC reader
- PR #3243 Use upstream join code in dask_cudf
- PR #3371 Add `select` method to `table_view`
- PR #3309 Add java and JNI bindings for search bounds
- PR #3305 Define and implement new rolling window APIs
- PR #3380 Concatenate columns of strings
- PR #3382 Add fill function for strings column
- PR #3391 Move device_atomics_tests.cu files to legacy
- PR #3303 Define and implement new stream compaction APIs `copy_if`, `drop_nulls`,
           `apply_boolean_mask`, `drop_duplicate` and `unique_count`.
- PR #3387 Strings column gather function
- PR #3440 Strings column scatter function
- PR #3389 Move quantiles.hpp + group_quantiles.hpp files to legacy
- PR #3397 Port unary cast to libcudf++
- PR #3398 Move reshape.hpp files to legacy
- PR #3423 Port NVStrings htoi to cudf strings column
- PR #3425 Strings column copy_if_else implementation
- PR #3422 Move utilities to legacy
- PR #3201 Define and implement new datetime_ops APIs
- PR #3421 Port NVStrings find_multiple to cudf strings column
- PR #3448 Port scatter_to_tables to libcudf++
- PR #3458 Update strings sections in the transition guide
- PR #3462 Add `make_empty_column` and update `empty_like`.
- PR #3465 Port `aggregation` traits and utilities.
- PR #3214 Define and implement new unary operations APIs
- PR #3475 Add `bitmask_to_host` column utility
- PR #3487 Add is_boolean trait and random timestamp generator for testing
- PR #3492 Small cleanup (remove std::abs) and comment
- PR #3407 Allow multiple row-groups per task in dask_cudf read_parquet
- PR #3512 Remove unused CUDA conda labels
- PR #3500 cudf::fill()/cudf::repeat() support for strings columns.
- PR #3438 Update scalar and scalar_device_view to better support strings
- PR #3414 Add copy_range function for strings column

## Bug Fixes

- PR #2895 Fixed dask_cudf group_split behavior to handle upstream rearrange_by_divisions
- PR #3048 Support for zero columned tables
- PR #3030 Fix snappy decoding regression in PR #3014
- PR #3041 Fixed exp to experimental namespace name change issue
- PR #3056 Add additional cmake hint for finding local build of RMM files
- PR #3060 Move copying.hpp includes to legacy
- PR #3139 Fixed java RMM auto initalization
- PR #3141 Java fix for relocated IO headers
- PR #3149 Rename column_wrapper.cuh to column_wrapper.hpp
- PR #3168 Fix mutable_column_device_view head const_cast
- PR #3199 Update JNI includes for legacy moves
- PR #3204 ORC writer: Fix ByteRLE encoding of NULLs
- PR #2994 Fix split_out-support but with hash_object_dispatch
- PR #3212 Fix string to date casting when format is not specified
- PR #3218 Fixes `row_lexicographic_comparator` issue with handling two tables
- PR #3228 Default initialize RMM when Java native dependencies are loaded
- PR #3012 replacing instances of `to_gpu_array` with `mem`
- PR #3236 Fix Numba 0.46+/CuPy 6.3 interface compatibility
- PR #3276 Update JNI includes for legacy moves
- PR #3256 Fix orc writer crash with multiple string columns
- PR #3211 Fix breaking change caused by rapidsai/rmm#167
- PR #3265 Fix dangling pointer in `is_sorted`
- PR #3267 ORC writer: fix incorrect ByteRLE encoding of long literal runs
- PR #3277 Fix invalid reference to deleted temporary in `is_sorted`.
- PR #3274 ORC writer: fix integer RLEv2 mode2 unsigned base value encoding
- PR #3279 Fix shutdown hang issues with pinned memory pool init executor
- PR #3280 Invalid children check in mutable_column_device_view
- PR #3289 fix java memory usage API for empty columns
- PR #3293 Fix loading of csv files zipped on MacOS (disabled zip min version check)
- PR #3295 Fix storing storing invalid RMM exec policies.
- PR #3307 Add pd.RangeIndex to from_pandas to fix dask_cudf meta_nonempty bug
- PR #3313 Fix public headers including non-public headers
- PR #3318 Revert arrow to 0.15.0 temporarily to unblock downstream projects CI
- PR #3317 Fix index-argument bug in dask_cudf parquet reader
- PR #3323 Fix `insert` non-assert test case
- PR #3341 Fix `Series` constructor converting NoneType to "None"
- PR #3326 Fix and test for detail::gather map iterator type inference
- PR #3334 Remove zero-size exception check from make_strings_column factories
- PR #3333 Fix compilation issues with `constexpr` functions not marked `__device__`
- PR #3340 Make all benchmarks use cudf base fixture to initialize RMM pool
- PR #3337 Fix Java to pad validity buffers to 64-byte boundary
- PR #3362 Fix `find_and_replace` upcasting series for python scalars and lists
- PR #3357 Disabling `column_view` iterators for non fixed-width types
- PR #3383 Fix : properly compute null counts for rolling_window.
- PR #3386 Removing external includes from `column_view.hpp`
- PR #3369 Add write_partition to dask_cudf to fix to_parquet bug
- PR #3388 Support getitem with bools when DataFrame has a MultiIndex
- PR #3408 Fix String and Column (De-)Serialization
- PR #3372 Fix dask-distributed scatter_by_map bug
- PR #3419 Fix a bug in parse_into_parts (incomplete input causing walking past the end of string).
- PR #3413 Fix dask_cudf read_csv file-list bug
- PR #3416 Fix memory leak in ColumnVector when pulling strings off the GPU
- PR #3424 Fix benchmark build by adding libcudacxx to benchmark's CMakeLists.txt
- PR #3435 Fix diff and shift for empty series
- PR #3439 Fix index-name bug in StringColumn concat
- PR #3445 Fix ORC Writer default stripe size
- PR #3459 Fix printing of invalid entries
- PR #3466 Fix gather null mask allocation for invalid index
- PR #3468 Fix memory leak issue in `drop_duplicates`
- PR #3474 Fix small doc error in capitalize Docs
- PR #3491 Fix more doc errors in NVStrings
- PR #3478 Fix as_index deep copy via Index.rename inplace arg
- PR #3476 Fix ORC reader timezone conversion
- PR #3188 Repr slices up large DataFrames
- PR #3519 Fix strings column concatenate handling zero-sized columns
- PR #3530 Fix copy_if_else test case fail issue
- PR #3523 Fix lgenfe issue with debug build
- PR #3532 Fix potential use-after-free in cudf parquet reader
- PR #3540 Fix unary_op null_mask bug and add missing test cases
- PR #3559 Use HighLevelGraph api in DataFrame constructor (Fix upstream compatibility)
- PR #3572 Fix CI Issue with hypothesis tests that are flaky


# cuDF 0.10.0 (16 Oct 2019)

## New Features

- PR #2423 Added `groupby.quantile()`
- PR #2522 Add Java bindings for NVStrings backed upper and lower case mutators
- PR #2605 Added Sort based groupby in libcudf
- PR #2607 Add Java bindings for parsing JSON
- PR #2629 Add dropna= parameter to groupby
- PR #2585 ORC & Parquet Readers: Remove millisecond timestamp restriction
- PR #2507 Add GPU-accelerated ORC Writer
- PR #2559 Add Series.tolist()
- PR #2653 Add Java bindings for rolling window operations
- PR #2480 Merge `custreamz` codebase into `cudf` repo
- PR #2674 Add __contains__ for Index/Series/Column
- PR #2635 Add support to read from remote and cloud sources like s3, gcs, hdfs
- PR #2722 Add Java bindings for NVTX ranges
- PR #2702 Add make_bool to dataset generation functions
- PR #2394 Move `rapidsai/custrings` into `cudf`
- PR #2734 Final sync of custrings source into cudf
- PR #2724 Add libcudf support for __contains__
- PR #2777 Add python bindings for porter stemmer measure functionality
- PR #2781 Add issorted to is_monotonic
- PR #2685 Add cudf::scatter_to_tables and cython binding
- PR #2743 Add Java bindings for NVStrings timestamp2long as part of String ColumnVector casting
- PR #2785 Add nvstrings Python docs
- PR #2786 Add benchmarks option to root build.sh
- PR #2802 Add `cudf::repeat()` and `cudf.Series.repeat()`
- PR #2773 Add Fisher's unbiased kurtosis and skew for Series/DataFrame
- PR #2748 Parquet Reader: Add option to specify loading of PANDAS index
- PR #2807 Add scatter_by_map to DataFrame python API
- PR #2836 Add nvstrings.code_points method
- PR #2844 Add Series/DataFrame notnull
- PR #2858 Add GTest type list utilities
- PR #2870 Add support for grouping by Series of arbitrary length
- PR #2719 Series covariance and Pearson correlation
- PR #2207 Beginning of libcudf overhaul: introduce new column and table types
- PR #2869 Add `cudf.CategoricalDtype`
- PR #2838 CSV Reader: Support ARROW_RANDOM_FILE input
- PR #2655 CuPy-based Series and Dataframe .values property
- PR #2803 Added `edit_distance_matrix()` function to calculate pairwise edit distance for each string on a given nvstrings object.
- PR #2811 Start of cudf strings column work based on 2207
- PR #2872 Add Java pinned memory pool allocator
- PR #2969 Add findAndReplaceAll to ColumnVector
- PR #2814 Add Datetimeindex.weekday
- PR #2999 Add timestamp conversion support for string categories
- PR #2918 Add cudf::column timestamp wrapper types

## Improvements

- PR #2578 Update legacy_groupby to use libcudf group_by_without_aggregation
- PR #2581 Removed `managed` allocator from hash map classes.
- PR #2571 Remove unnecessary managed memory from gdf_column_concat
- PR #2648 Cython/Python reorg
- PR #2588 Update Series.append documentation
- PR #2632 Replace dask-cudf set_index code with upstream
- PR #2682 Add cudf.set_allocator() function for easier allocator init
- PR #2642 Improve null printing and testing
- PR #2747 Add missing Cython headers / cudftestutil lib to conda package for cuspatial build
- PR #2706 Compute CSV format in device code to speedup performance
- PR #2673 Add support for np.longlong type
- PR #2703 move dask serialization dispatch into cudf
- PR #2728 Add YYMMDD to version tag for nightly conda packages
- PR #2729 Handle file-handle input in to_csv
- PR #2741 CSV Reader: Move kernel functions into its own file
- PR #2766 Improve nvstrings python cmake flexibility
- PR #2756 Add out_time_unit option to csv reader, support timestamp resolutions
- PR #2771 Stopgap alias for to_gpu_matrix()
- PR #2783 Support mapping input columns to function arguments in apply kernels
- PR #2645 libcudf unique_count for Series.nunique
- PR #2817 Dask-cudf: `read_parquet` support for remote filesystems
- PR #2823 improve java data movement debugging
- PR #2806 CSV Reader: Clean-up row offset operations
- PR #2640 Add dask wait/persist exmaple to 10 minute guide
- PR #2828 Optimizations of kernel launch configuration for `DataFrame.apply_rows` and `DataFrame.apply_chunks`
- PR #2831 Add `column` argument to `DataFrame.drop`
- PR #2775 Various optimizations to improve __getitem__ and __setitem__ performance
- PR #2810 cudf::allocate_like can optionally always allocate a mask.
- PR #2833 Parquet reader: align page data allocation sizes to 4-bytes to satisfy cuda-memcheck
- PR #2832 Using the new Python bindings for UCX
- PR #2856 Update group_split_cudf to use scatter_by_map
- PR #2890 Optionally keep serialized table data on the host.
- PR #2778 Doc: Updated and fixed some docstrings that were formatted incorrectly.
- PR #2830 Use YYMMDD tag in custreamz nightly build
- PR #2875 Java: Remove synchronized from register methods in MemoryCleaner
- PR #2887 Minor snappy decompression optimization
- PR #2899 Use new RMM API based on Cython
- PR #2788 Guide to Python UDFs
- PR #2919 Change java API to use operators in groupby namespace
- PR #2909 CSV Reader: Avoid row offsets host vector default init
- PR #2834 DataFrame supports setting columns via attribute syntax `df.x = col`
- PR #3147 DataFrame can be initialized from rows via list of tuples
- PR #3539 Restrict CuPy to 6

## Bug Fixes

- PR #2584 ORC Reader: fix parsing of `DECIMAL` index positions
- PR #2619 Fix groupby serialization/deserialization
- PR #2614 Update Java version to match
- PR #2601 Fixes nlargest(1) issue in Series and Dataframe
- PR #2610 Fix a bug in index serialization (properly pass DeviceNDArray)
- PR #2621 Fixes the floordiv issue of not promoting float type when rhs is 0
- PR #2611 Types Test: fix static casting from negative int to string
- PR #2618 IO Readers: Fix datasource memory map failure for multiple reads
- PR #2628 groupby_without_aggregation non-nullable input table produces non-nullable output
- PR #2615 fix string category partitioning in java API
- PR #2641 fix string category and timeunit concat in the java API
- PR #2649 Fix groupby issue resulting from column_empty bug
- PR #2658 Fix astype() for null categorical columns
- PR #2660 fix column string category and timeunit concat in the java API
- PR #2664 ORC reader: fix `skip_rows` larger than first stripe
- PR #2654 Allow Java gdfOrderBy to work with string categories
- PR #2669 AVRO reader: fix non-deterministic output
- PR #2668 Update Java bindings to specify timestamp units for ORC and Parquet readers
- PR #2679 AVRO reader: fix cuda errors when decoding compressed streams
- PR #2692 Add concatenation for data-frame with different headers (empty and non-empty)
- PR #2651 Remove nvidia driver installation from ci/cpu/build.sh
- PR #2697 Ensure csv reader sets datetime column time units
- PR #2698 Return RangeIndex from contiguous slice of RangeIndex
- PR #2672 Fix null and integer handling in round
- PR #2704 Parquet Reader: Fix crash when loading string column with nulls
- PR #2725 Fix Jitify issue with running on Turing using CUDA version < 10
- PR #2731 Fix building of benchmarks
- PR #2738 Fix java to find new NVStrings locations
- PR #2736 Pin Jitify branch to v0.10 version
- PR #2742 IO Readers: Fix possible silent failures when creating `NvStrings` instance
- PR #2753 Fix java quantile API calls
- PR #2762 Fix validity processing for time in java
- PR #2796 Fix handling string slicing and other nvstrings delegated methods with dask
- PR #2769 Fix link to API docs in README.md
- PR #2772 Handle multiindex pandas Series #2772
- PR #2749 Fix apply_rows/apply_chunks pessimistic null mask to use in_cols null masks only
- PR #2752 CSV Reader: Fix exception when there's no rows to process
- PR #2716 Added Exception for `StringMethods` in string methods
- PR #2787 Fix Broadcasting `None` to `cudf-series`
- PR #2794 Fix async race in NVCategory::get_value and get_value_bounds
- PR #2795 Fix java build/cast error
- PR #2496 Fix improper merge of two dataframes when names differ
- PR #2824 Fix issue with incorrect result when Numeric Series replace is called several times
- PR #2751 Replace value with null
- PR #2765 Fix Java inequality comparisons for string category
- PR #2818 Fix java join API to use new C++ join API
- PR #2841 Fix nvstrings.slice and slice_from for range (0,0)
- PR #2837 Fix join benchmark
- PR #2809 Add hash_df and group_split dispatch functions for dask
- PR #2843 Parquet reader: fix skip_rows when not aligned with page or row_group boundaries
- PR #2851 Deleted existing dask-cudf/record.txt
- PR #2854 Fix column creation from ephemeral objects exposing __cuda_array_interface__
- PR #2860 Fix boolean indexing when the result is a single row
- PR #2859 Fix tail method issue for string columns
- PR #2852 Fixed `cumsum()` and `cumprod()` on boolean series.
- PR #2865 DaskIO: Fix `read_csv` and `read_orc` when input is list of files
- PR #2750 Fixed casting values to cudf::bool8 so non-zero values always cast to true
- PR #2873 Fixed dask_cudf read_partition bug by generating ParquetDatasetPiece
- PR #2850 Fixes dask_cudf.read_parquet on partitioned datasets
- PR #2896 Properly handle `axis` string keywords in `concat`
- PR #2926 Update rounding algorithm to avoid using fmod
- PR #2968 Fix Java dependency loading when using NVTX
- PR #2963 Fix ORC writer uncompressed block indexing
- PR #2928 CSV Reader: Fix using `byte_range` for large datasets
- PR #2983 Fix sm_70+ race condition in gpu_unsnap
- PR #2964 ORC Writer: Segfault when writing mixed numeric and string columns
- PR #3007 Java: Remove unit test that frees RMM invalid pointer
- PR #3009 Fix orc reader RLEv2 patch position regression from PR #2507
- PR #3002 Fix CUDA invalid configuration errors reported after loading an ORC file without data
- PR #3035 Update update-version.sh for new docs locations
- PR #3038 Fix uninitialized stream parameter in device_table deleter
- PR #3064 Fixes groupby performance issue
- PR #3061 Add rmmInitialize to nvstrings gtests
- PR #3058 Fix UDF doc markdown formatting
- PR #3059 Add nvstrings python build instructions to contributing.md


# cuDF 0.9.0 (21 Aug 2019)

## New Features

- PR #1993 Add CUDA-accelerated series aggregations: mean, var, std
- PR #2111 IO Readers: Support memory buffer, file-like object, and URL inputs
- PR #2012 Add `reindex()` to DataFrame and Series
- PR #2097 Add GPU-accelerated AVRO reader
- PR #2098 Support binary ops on DFs and Series with mismatched indices
- PR #2160 Merge `dask-cudf` codebase into `cudf` repo
- PR #2149 CSV Reader: Add `hex` dtype for explicit hexadecimal parsing
- PR #2156 Add `upper_bound()` and `lower_bound()` for libcudf tables and `searchsorted()` for cuDF Series
- PR #2158 CSV Reader: Support single, non-list/dict argument for `dtype`
- PR #2177 CSV Reader: Add `parse_dates` parameter for explicit date inference
- PR #1744 cudf::apply_boolean_mask and cudf::drop_nulls support for cudf::table inputs (multi-column)
- PR #2196 Add `DataFrame.dropna()`
- PR #2197 CSV Writer: add `chunksize` parameter for `to_csv`
- PR #2215 `type_dispatcher` benchmark
- PR #2179 Add Java quantiles
- PR #2157 Add __array_function__ to DataFrame and Series
- PR #2212 Java support for ORC reader
- PR #2224 Add DataFrame isna, isnull, notna functions
- PR #2236 Add Series.drop_duplicates
- PR #2105 Add hash-based join benchmark
- PR #2316 Add unique, nunique, and value_counts for datetime columns
- PR #2337 Add Java support for slicing a ColumnVector
- PR #2049 Add cudf::merge (sorted merge)
- PR #2368 Full cudf+dask Parquet Support
- PR #2380 New cudf::is_sorted checks whether cudf::table is sorted
- PR #2356 Java column vector standard deviation support
- PR #2221 MultiIndex full indexing - Support iloc and wildcards for loc
- PR #2429 Java support for getting length of strings in a ColumnVector
- PR #2415 Add `value_counts` for series of any type
- PR #2446 Add __array_function__ for index
- PR #2437 ORC reader: Add 'use_np_dtypes' option
- PR #2382 Add CategoricalAccessor add, remove, rename, and ordering methods
- PR #2464 Native implement `__cuda_array_interface__` for Series/Index/Column objects
- PR #2425 Rolling window now accepts array-based user-defined functions
- PR #2442 Add __setitem__
- PR #2449 Java support for getting byte count of strings in a ColumnVector
- PR #2492 Add groupby.size() method
- PR #2358 Add cudf::nans_to_nulls: convert floating point column into bitmask
- PR #2489 Add drop argument to set_index
- PR #2491 Add Java bindings for ORC reader 'use_np_dtypes' option
- PR #2213 Support s/ms/us/ns DatetimeColumn time unit resolutions
- PR #2536 Add _constructor properties to Series and DataFrame

## Improvements

- PR #2103 Move old `column` and `bitmask` files into `legacy/` directory
- PR #2109 added name to Python column classes
- PR #1947 Cleanup serialization code
- PR #2125 More aggregate in java API
- PR #2127 Add in java Scalar tests
- PR #2088 Refactor of Python groupby code
- PR #2130 Java serialization and deserialization of tables.
- PR #2131 Chunk rows logic added to csv_writer
- PR #2129 Add functions in the Java API to support nullable column filtering
- PR #2165 made changes to get_dummies api for it to be available in MethodCache
- PR #2171 Add CodeCov integration, fix doc version, make --skip-tests work when invoking with source
- PR #2184 handle remote orc files for dask-cudf
- PR #2186 Add `getitem` and `getattr` style access to Rolling objects
- PR #2168 Use cudf.Column for CategoricalColumn's categories instead of a tuple
- PR #2193 DOC: cudf::type_dispatcher documentation for specializing dispatched functors
- PR #2199 Better java support for appending strings
- PR #2176 Added column dtype support for datetime, int8, int16 to csv_writer
- PR #2209 Matching `get_dummies` & `select_dtypes` behavior to pandas
- PR #2217 Updated Java bindings to use the new groupby API
- PR #2214 DOC: Update doc instructions to build/install `cudf` and `dask-cudf`
- PR #2220 Update Java bindings for reduction rename
- PR #2232 Move CodeCov upload from build script to Jenkins
- PR #2225 refactor to use libcudf for gathering columns in dataframes
- PR #2293 Improve join performance (faster compute_join_output_size)
- PR #2300 Create separate dask codeowners for dask-cudf codebase
- PR #2304 gdf_group_by_without_aggregations returns gdf_column
- PR #2309 Java readers: remove redundant copy of result pointers
- PR #2307 Add `black` and `isort` to style checker script
- PR #2345 Restore removal of old groupby implementation
- PR #2342 Improve `astype()` to operate all ways
- PR #2329 using libcudf cudf::copy for column deep copy
- PR #2344 DOC: docs on code formatting for contributors
- PR #2376 Add inoperative axis= and win_type= arguments to Rolling()
- PR #2378 remove dask for (de-)serialization of cudf objects
- PR #2353 Bump Arrow and Dask versions
- PR #2377 Replace `standard_python_slice` with just `slice.indices()`
- PR #2373 cudf.DataFrame enchancements & Series.values support
- PR #2392 Remove dlpack submodule; make cuDF's Cython API externally accessible
- PR #2430 Updated Java bindings to use the new unary API
- PR #2406 Moved all existing `table` related files to a `legacy/` directory
- PR #2350 Performance related changes to get_dummies
- PR #2420 Remove `cudautils.astype` and replace with `typecast.apply_cast`
- PR #2456 Small improvement to typecast utility
- PR #2458 Fix handling of thirdparty packages in `isort` config
- PR #2459 IO Readers: Consolidate all readers to use `datasource` class
- PR #2475 Exposed type_dispatcher.hpp, nvcategory_util.hpp and wrapper_types.hpp in the include folder
- PR #2484 Enabled building libcudf as a static library
- PR #2453 Streamline CUDA_REL environment variable
- PR #2483 Bundle Boost filesystem dependency in the Java jar
- PR #2486 Java API hash functions
- PR #2481 Adds the ignore_null_keys option to the java api
- PR #2490 Java api: support multiple aggregates for the same column
- PR #2510 Java api: uses table based apply_boolean_mask
- PR #2432 Use pandas formatting for console, html, and latex output
- PR #2573 Bump numba version to 0.45.1
- PR #2606 Fix references to notebooks-contrib

## Bug Fixes

- PR #2086 Fixed quantile api behavior mismatch in series & dataframe
- PR #2128 Add offset param to host buffer readers in java API.
- PR #2145 Work around binops validity checks for java
- PR #2146 Work around unary_math validity checks for java
- PR #2151 Fixes bug in cudf::copy_range where null_count was invalid
- PR #2139 matching to pandas describe behavior & fixing nan values issue
- PR #2161 Implicitly convert unsigned to signed integer types in binops
- PR #2154 CSV Reader: Fix bools misdetected as strings dtype
- PR #2178 Fix bug in rolling bindings where a view of an ephemeral column was being taken
- PR #2180 Fix issue with isort reordering `importorskip` below imports depending on them
- PR #2187 fix to honor dtype when numpy arrays are passed to columnops.as_column
- PR #2190 Fix issue in astype conversion of string column to 'str'
- PR #2208 Fix issue with calling `head()` on one row dataframe
- PR #2229 Propagate exceptions from Cython cdef functions
- PR #2234 Fix issue with local build script not properly building
- PR #2223 Fix CUDA invalid configuration errors reported after loading small compressed ORC files
- PR #2162 Setting is_unique and is_monotonic-related attributes
- PR #2244 Fix ORC RLEv2 delta mode decoding with nonzero residual delta width
- PR #2297 Work around `var/std` unsupported only at debug build
- PR #2302 Fixed java serialization corner case
- PR #2355 Handle float16 in binary operations
- PR #2311 Fix copy behaviour for GenericIndex
- PR #2349 Fix issues with String filter in java API
- PR #2323 Fix groupby on categoricals
- PR #2328 Ensure order is preserved in CategoricalAccessor._set_categories
- PR #2202 Fix issue with unary ops mishandling empty input
- PR #2326 Fix for bug in DLPack when reading multiple columns
- PR #2324 Fix cudf Docker build
- PR #2325 Fix ORC RLEv2 patched base mode decoding with nonzero patch width
- PR #2235 Fix get_dummies to be compatible with dask
- PR #2332 Zero initialize gdf_dtype_extra_info
- PR #2355 Handle float16 in binary operations
- PR #2360 Fix missing dtype handling in cudf.Series & columnops.as_column
- PR #2364 Fix quantile api and other trivial issues around it
- PR #2361 Fixed issue with `codes` of CategoricalIndex
- PR #2357 Fixed inconsistent type of index created with from_pandas vs direct construction
- PR #2389 Fixed Rolling __getattr__ and __getitem__ for offset based windows
- PR #2402 Fixed bug in valid mask computation in cudf::copy_if (apply_boolean_mask)
- PR #2401 Fix to a scalar datetime(of type Days) issue
- PR #2386 Correctly allocate output valids in groupby
- PR #2411 Fixed failures on binary op on single element string column
- PR #2422 Fix Pandas logical binary operation incompatibilites
- PR #2447 Fix CodeCov posting build statuses temporarily
- PR #2450 Fix erroneous null handling in `cudf.DataFrame`'s `apply_rows`
- PR #2470 Fix issues with empty strings and string categories (Java)
- PR #2471 Fix String Column Validity.
- PR #2481 Fix java validity buffer serialization
- PR #2485 Updated bytes calculation to use size_t to avoid overflow in column concat
- PR #2461 Fix groupby multiple aggregations same column
- PR #2514 Fix cudf::drop_nulls threshold handling in Cython
- PR #2516 Fix utilities include paths and meta.yaml header paths
- PR #2517 Fix device memory leak in to_dlpack tensor deleter
- PR #2431 Fix local build generated file ownerships
- PR #2511 Added import of orc, refactored exception handlers to not squash fatal exceptions
- PR #2527 Fix index and column input handling in dask_cudf read_parquet
- PR #2466 Fix `dataframe.query` returning null rows erroneously
- PR #2548 Orc reader: fix non-deterministic data decoding at chunk boundaries
- PR #2557 fix cudautils import in string.py
- PR #2521 Fix casting datetimes from/to the same resolution
- PR #2545 Fix MultiIndexes with datetime levels
- PR #2560 Remove duplicate `dlpack` definition in conda recipe
- PR #2567 Fix ColumnVector.fromScalar issues while dealing with null scalars
- PR #2565 Orc reader: fix incorrect data decoding of int64 data types
- PR #2577 Fix search benchmark compilation error by adding necessary header
- PR #2604 Fix a bug in copying.pyx:_normalize_types that upcasted int32 to int64


# cuDF 0.8.0 (27 June 2019)

## New Features

- PR #1524 Add GPU-accelerated JSON Lines parser with limited feature set
- PR #1569 Add support for Json objects to the JSON Lines reader
- PR #1622 Add Series.loc
- PR #1654 Add cudf::apply_boolean_mask: faster replacement for gdf_apply_stencil
- PR #1487 cython gather/scatter
- PR #1310 Implemented the slice/split functionality.
- PR #1630 Add Python layer to the GPU-accelerated JSON reader
- PR #1745 Add rounding of numeric columns via Numba
- PR #1772 JSON reader: add support for BytesIO and StringIO input
- PR #1527 Support GDF_BOOL8 in readers and writers
- PR #1819 Logical operators (AND, OR, NOT) for libcudf and cuDF
- PR #1813 ORC Reader: Add support for stripe selection
- PR #1828 JSON Reader: add suport for bool8 columns
- PR #1833 Add column iterator with/without nulls
- PR #1665 Add the point-in-polygon GIS function
- PR #1863 Series and Dataframe methods for all and any
- PR #1908 cudf::copy_range and cudf::fill for copying/assigning an index or range to a constant
- PR #1921 Add additional formats for typecasting to/from strings
- PR #1807 Add Series.dropna()
- PR #1987 Allow user defined functions in the form of ptx code to be passed to binops
- PR #1948 Add operator functions like `Series.add()` to DataFrame and Series
- PR #1954 Add skip test argument to GPU build script
- PR #2018 Add bindings for new groupby C++ API
- PR #1984 Add rolling window operations Series.rolling() and DataFrame.rolling()
- PR #1542 Python method and bindings for to_csv
- PR #1995 Add Java API
- PR #1998 Add google benchmark to cudf
- PR #1845 Add cudf::drop_duplicates, DataFrame.drop_duplicates
- PR #1652 Added `Series.where()` feature
- PR #2074 Java Aggregates, logical ops, and better RMM support
- PR #2140 Add a `cudf::transform` function
- PR #2068 Concatenation of different typed columns

## Improvements

- PR #1538 Replacing LesserRTTI with inequality_comparator
- PR #1703 C++: Added non-aggregating `insert` to `concurrent_unordered_map` with specializations to store pairs with a single atomicCAS when possible.
- PR #1422 C++: Added a RAII wrapper for CUDA streams
- PR #1701 Added `unique` method for stringColumns
- PR #1713 Add documentation for Dask-XGBoost
- PR #1666 CSV Reader: Improve performance for files with large number of columns
- PR #1725 Enable the ability to use a single column groupby as its own index
- PR #1759 Add an example showing simultaneous rolling averages to `apply_grouped` documentation
- PR #1746 C++: Remove unused code: `windowed_ops.cu`, `sorting.cu`, `hash_ops.cu`
- PR #1748 C++: Add `bool` nullability flag to `device_table` row operators
- PR #1764 Improve Numerical column: `mean_var` and `mean`
- PR #1767 Speed up Python unit tests
- PR #1770 Added build.sh script, updated CI scripts and documentation
- PR #1739 ORC Reader: Add more pytest coverage
- PR #1696 Added null support in `Series.replace()`.
- PR #1390 Added some basic utility functions for `gdf_column`'s
- PR #1791 Added general column comparison code for testing
- PR #1795 Add printing of git submodule info to `print_env.sh`
- PR #1796 Removing old sort based group by code and gdf_filter
- PR #1811 Added funtions for copying/allocating `cudf::table`s
- PR #1838 Improve columnops.column_empty so that it returns typed columns instead of a generic Column
- PR #1890 Add utils.get_dummies- a pandas-like wrapper around one_hot-encoding
- PR #1823 CSV Reader: default the column type to string for empty dataframes
- PR #1827 Create bindings for scalar-vector binops, and update one_hot_encoding to use them
- PR #1817 Operators now support different sized dataframes as long as they don't share different sized columns
- PR #1855 Transition replace_nulls to new C++ API and update corresponding Cython/Python code
- PR #1858 Add `std::initializer_list` constructor to `column_wrapper`
- PR #1846 C++ type-erased gdf_equal_columns test util; fix gdf_equal_columns logic error
- PR #1390 Added some basic utility functions for `gdf_column`s
- PR #1391 Tidy up bit-resolution-operation and bitmask class code
- PR #1882 Add iloc functionality to MultiIndex dataframes
- PR #1884 Rolling windows: general enhancements and better coverage for unit tests
- PR #1886 support GDF_STRING_CATEGORY columns in apply_boolean_mask, drop_nulls and other libcudf functions
- PR #1896 Improve performance of groupby with levels specified in dask-cudf
- PR #1915 Improve iloc performance for non-contiguous row selection
- PR #1859 Convert read_json into a C++ API
- PR #1919 Rename libcudf namespace gdf to namespace cudf
- PR #1850 Support left_on and right_on for DataFrame merge operator
- PR #1930 Specialize constructor for `cudf::bool8` to cast argument to `bool`
- PR #1938 Add default constructor for `column_wrapper`
- PR #1930 Specialize constructor for `cudf::bool8` to cast argument to `bool`
- PR #1952 consolidate libcudf public API headers in include/cudf
- PR #1949 Improved selection with boolmask using libcudf `apply_boolean_mask`
- PR #1956 Add support for nulls in `query()`
- PR #1973 Update `std::tuple` to `std::pair` in top-most libcudf APIs and C++ transition guide
- PR #1981 Convert read_csv into a C++ API
- PR #1868 ORC Reader: Support row index for speed up on small/medium datasets
- PR #1964 Added support for list-like types in Series.str.cat
- PR #2005 Use HTML5 details tag in bug report issue template
- PR #2003 Removed few redundant unit-tests from test_string.py::test_string_cat
- PR #1944 Groupby design improvements
- PR #2017 Convert `read_orc()` into a C++ API
- PR #2011 Convert `read_parquet()` into a C++ API
- PR #1756 Add documentation "10 Minutes to cuDF and dask_cuDF"
- PR #2034 Adding support for string columns concatenation using "add" binary operator
- PR #2042 Replace old "10 Minutes" guide with new guide for docs build process
- PR #2036 Make library of common test utils to speed up tests compilation
- PR #2022 Facilitating get_dummies to be a high level api too
- PR #2050 Namespace IO readers and add back free-form `read_xxx` functions
- PR #2104 Add a functional ``sort=`` keyword argument to groupby
- PR #2108 Add `find_and_replace` for StringColumn for replacing single values
- PR #1803 cuDF/CuPy interoperability documentation

## Bug Fixes

- PR #1465 Fix for test_orc.py and test_sparse_df.py test failures
- PR #1583 Fix underlying issue in `as_index()` that was causing `Series.quantile()` to fail
- PR #1680 Add errors= keyword to drop() to fix cudf-dask bug
- PR #1651 Fix `query` function on empty dataframe
- PR #1616 Fix CategoricalColumn to access categories by index instead of iteration
- PR #1660 Fix bug in `loc` when indexing with a column name (a string)
- PR #1683 ORC reader: fix timestamp conversion to UTC
- PR #1613 Improve CategoricalColumn.fillna(-1) performance
- PR #1642 Fix failure of CSV_TEST gdf_csv_test.SkiprowsNrows on multiuser systems
- PR #1709 Fix handling of `datetime64[ms]` in `dataframe.select_dtypes`
- PR #1704 CSV Reader: Add support for the plus sign in number fields
- PR #1687 CSV reader: return an empty dataframe for zero size input
- PR #1757 Concatenating columns with null columns
- PR #1755 Add col_level keyword argument to melt
- PR #1758 Fix df.set_index() when setting index from an empty column
- PR #1749 ORC reader: fix long strings of NULL values resulting in incorrect data
- PR #1742 Parquet Reader: Fix index column name to match PANDAS compat
- PR #1782 Update libcudf doc version
- PR #1783 Update conda dependencies
- PR #1786 Maintain the original series name in series.unique output
- PR #1760 CSV Reader: fix segfault when dtype list only includes columns from usecols list
- PR #1831 build.sh: Assuming python is in PATH instead of using PYTHON env var
- PR #1839 Raise an error instead of segfaulting when transposing a DataFrame with StringColumns
- PR #1840 Retain index correctly during merge left_on right_on
- PR #1825 cuDF: Multiaggregation Groupby Failures
- PR #1789 CSV Reader: Fix missing support for specifying `int8` and `int16` dtypes
- PR #1857 Cython Bindings: Handle `bool` columns while calling `column_view_from_NDArrays`
- PR #1849 Allow DataFrame support methods to pass arguments to the methods
- PR #1847 Fixed #1375 by moving the nvstring check into the wrapper function
- PR #1864 Fixing cudf reduction for POWER platform
- PR #1869 Parquet reader: fix Dask timestamps not matching with Pandas (convert to milliseconds)
- PR #1876 add dtype=bool for `any`, `all` to treat integer column correctly
- PR #1875 CSV reader: take NaN values into account in dtype detection
- PR #1873 Add column dtype checking for the all/any methods
- PR #1902 Bug with string iteration in _apply_basic_agg
- PR #1887 Fix for initialization issue in pq_read_arg,orc_read_arg
- PR #1867 JSON reader: add support for null/empty fields, including the 'null' literal
- PR #1891 Fix bug #1750 in string column comparison
- PR #1909 Support of `to_pandas()` of boolean series with null values
- PR #1923 Use prefix removal when two aggs are called on a SeriesGroupBy
- PR #1914 Zero initialize gdf_column local variables
- PR #1959 Add support for comparing boolean Series to scalar
- PR #1966 Ignore index fix in series append
- PR #1967 Compute index __sizeof__ only once for DataFrame __sizeof__
- PR #1977 Support CUDA installation in default system directories
- PR #1982 Fixes incorrect index name after join operation
- PR #1985 Implement `GDF_PYMOD`, a special modulo that follows python's sign rules
- PR #1991 Parquet reader: fix decoding of NULLs
- PR #1990 Fixes a rendering bug in the `apply_grouped` documentation
- PR #1978 Fix for values being filled in an empty dataframe
- PR #2001 Correctly create MultiColumn from Pandas MultiColumn
- PR #2006 Handle empty dataframe groupby construction for dask
- PR #1965 Parquet Reader: Fix duplicate index column when it's already in `use_cols`
- PR #2033 Add pip to conda environment files to fix warning
- PR #2028 CSV Reader: Fix reading of uncompressed files without a recognized file extension
- PR #2073 Fix an issue when gathering columns with NVCategory and nulls
- PR #2053 cudf::apply_boolean_mask return empty column for empty boolean mask
- PR #2066 exclude `IteratorTest.mean_var_output` test from debug build
- PR #2069 Fix JNI code to use read_csv and read_parquet APIs
- PR #2071 Fix bug with unfound transitive dependencies for GTests in Ubuntu 18.04
- PR #2089 Configure Sphinx to render params correctly
- PR #2091 Fix another bug with unfound transitive dependencies for `cudftestutils` in Ubuntu 18.04
- PR #2115 Just apply `--disable-new-dtags` instead of trying to define all the transitive dependencies
- PR #2106 Fix errors in JitCache tests caused by sharing of device memory between processes
- PR #2120 Fix errors in JitCache tests caused by running multiple threads on the same data
- PR #2102 Fix memory leak in groupby
- PR #2113 fixed typo in to_csv code example


# cudf 0.7.2 (16 May 2019)

## New Features

- PR #1735 Added overload for atomicAdd on int64. Streamlined implementation of custom atomic overloads.
- PR #1741 Add MultiIndex concatenation

## Bug Fixes

- PR #1718 Fix issue with SeriesGroupBy MultiIndex in dask-cudf
- PR #1734 Python: fix performance regression for groupby count() aggregations
- PR #1768 Cython: fix handling read only schema buffers in gpuarrow reader


# cudf 0.7.1 (11 May 2019)

## New Features

- PR #1702 Lazy load MultiIndex to return groupby performance to near optimal.

## Bug Fixes

- PR #1708 Fix handling of `datetime64[ms]` in `dataframe.select_dtypes`


# cuDF 0.7.0 (10 May 2019)

## New Features

- PR #982 Implement gdf_group_by_without_aggregations and gdf_unique_indices functions
- PR #1142 Add `GDF_BOOL` column type
- PR #1194 Implement overloads for CUDA atomic operations
- PR #1292 Implemented Bitwise binary ops AND, OR, XOR (&, |, ^)
- PR #1235 Add GPU-accelerated Parquet Reader
- PR #1335 Added local_dict arg in `DataFrame.query()`.
- PR #1282 Add Series and DataFrame.describe()
- PR #1356 Rolling windows
- PR #1381 Add DataFrame._get_numeric_data
- PR #1388 Add CODEOWNERS file to auto-request reviews based on where changes are made
- PR #1396 Add DataFrame.drop method
- PR #1413 Add DataFrame.melt method
- PR #1412 Add DataFrame.pop()
- PR #1419 Initial CSV writer function
- PR #1441 Add Series level cumulative ops (cumsum, cummin, cummax, cumprod)
- PR #1420 Add script to build and test on a local gpuCI image
- PR #1440 Add DatetimeColumn.min(), DatetimeColumn.max()
- PR #1455 Add Series.Shift via Numba kernel
- PR #1441 Add Series level cumulative ops (cumsum, cummin, cummax, cumprod)
- PR #1461 Add Python coverage test to gpu build
- PR #1445 Parquet Reader: Add selective reading of rows and row group
- PR #1532 Parquet Reader: Add support for INT96 timestamps
- PR #1516 Add Series and DataFrame.ndim
- PR #1556 Add libcudf C++ transition guide
- PR #1466 Add GPU-accelerated ORC Reader
- PR #1565 Add build script for nightly doc builds
- PR #1508 Add Series isna, isnull, and notna
- PR #1456 Add Series.diff() via Numba kernel
- PR #1588 Add Index `astype` typecasting
- PR #1301 MultiIndex support
- PR #1599 Level keyword supported in groupby
- PR #929 Add support operations to dataframe
- PR #1609 Groupby accept list of Series
- PR #1658 Support `group_keys=True` keyword in groupby method

## Improvements

- PR #1531 Refactor closures as private functions in gpuarrow
- PR #1404 Parquet reader page data decoding speedup
- PR #1076 Use `type_dispatcher` in join, quantiles, filter, segmented sort, radix sort and hash_groupby
- PR #1202 Simplify README.md
- PR #1149 CSV Reader: Change convertStrToValue() functions to `__device__` only
- PR #1238 Improve performance of the CUDA trie used in the CSV reader
- PR #1245 Use file cache for JIT kernels
- PR #1278 Update CONTRIBUTING for new conda environment yml naming conventions
- PR #1163 Refactored UnaryOps. Reduced API to two functions: `gdf_unary_math` and `gdf_cast`. Added `abs`, `-`, and `~` ops. Changed bindings to Cython
- PR #1284 Update docs version
- PR #1287 add exclude argument to cudf.select_dtype function
- PR #1286 Refactor some of the CSV Reader kernels into generic utility functions
- PR #1291 fillna in `Series.to_gpu_array()` and `Series.to_array()` can accept the scalar too now.
- PR #1005 generic `reduction` and `scan` support
- PR #1349 Replace modernGPU sort join with thrust.
- PR #1363 Add a dataframe.mean(...) that raises NotImplementedError to satisfy `dask.dataframe.utils.is_dataframe_like`
- PR #1319 CSV Reader: Use column wrapper for gdf_column output alloc/dealloc
- PR #1376 Change series quantile default to linear
- PR #1399 Replace CFFI bindings for NVTX functions with Cython bindings
- PR #1389 Refactored `set_null_count()`
- PR #1386 Added macros `GDF_TRY()`, `CUDF_TRY()` and `ASSERT_CUDF_SUCCEEDED()`
- PR #1435 Rework CMake and conda recipes to depend on installed libraries
- PR #1391 Tidy up bit-resolution-operation and bitmask class code
- PR #1439 Add cmake variable to enable compiling CUDA code with -lineinfo
- PR #1462 Add ability to read parquet files from arrow::io::RandomAccessFile
- PR #1453 Convert CSV Reader CFFI to Cython
- PR #1479 Convert Parquet Reader CFFI to Cython
- PR #1397 Add a utility function for producing an overflow-safe kernel launch grid configuration
- PR #1382 Add GPU parsing of nested brackets to cuIO parsing utilities
- PR #1481 Add cudf::table constructor to allocate a set of `gdf_column`s
- PR #1484 Convert GroupBy CFFI to Cython
- PR #1463 Allow and default melt keyword argument var_name to be None
- PR #1486 Parquet Reader: Use device_buffer rather than device_ptr
- PR #1525 Add cudatoolkit conda dependency
- PR #1520 Renamed `src/dataframe` to `src/table` and moved `table.hpp`. Made `types.hpp` to be type declarations only.
- PR #1492 Convert transpose CFFI to Cython
- PR #1495 Convert binary and unary ops CFFI to Cython
- PR #1503 Convert sorting and hashing ops CFFI to Cython
- PR #1522 Use latest release version in update-version CI script
- PR #1533 Remove stale join CFFI, fix memory leaks in join Cython
- PR #1521 Added `row_bitmask` to compute bitmask for rows of a table. Merged `valids_ops.cu` and `bitmask_ops.cu`
- PR #1553 Overload `hash_row` to avoid using intial hash values. Updated `gdf_hash` to select between overloads
- PR #1585 Updated `cudf::table` to maintain own copy of wrapped `gdf_column*`s
- PR #1559 Add `except +` to all Cython function definitions to catch C++ exceptions properly
- PR #1617 `has_nulls` and `column_dtypes` for `cudf::table`
- PR #1590 Remove CFFI from the build / install process entirely
- PR #1536 Convert gpuarrow CFFI to Cython
- PR #1655 Add `Column._pointer` as a way to access underlying `gdf_column*` of a `Column`
- PR #1655 Update readme conda install instructions for cudf version 0.6 and 0.7


## Bug Fixes

- PR #1233 Fix dtypes issue while adding the column to `str` dataframe.
- PR #1254 CSV Reader: fix data type detection for floating-point numbers in scientific notation
- PR #1289 Fix looping over each value instead of each category in concatenation
- PR #1293 Fix Inaccurate error message in join.pyx
- PR #1308 Add atomicCAS overload for `int8_t`, `int16_t`
- PR #1317 Fix catch polymorphic exception by reference in ipc.cu
- PR #1325 Fix dtype of null bitmasks to int8
- PR #1326 Update build documentation to use -DCMAKE_CXX11_ABI=ON
- PR #1334 Add "na_position" argument to CategoricalColumn sort_by_values
- PR #1321 Fix out of bounds warning when checking Bzip2 header
- PR #1359 Add atomicAnd/Or/Xor for integers
- PR #1354 Fix `fillna()` behaviour when replacing values with different dtypes
- PR #1347 Fixed core dump issue while passing dict_dtypes without column names in `cudf.read_csv()`
- PR #1379 Fixed build failure caused due to error: 'col_dtype' may be used uninitialized
- PR #1392 Update cudf Dockerfile and package_versions.sh
- PR #1385 Added INT8 type to `_schema_to_dtype` for use in GpuArrowReader
- PR #1393 Fixed a bug in `gdf_count_nonzero_mask()` for the case of 0 bits to count
- PR #1395 Update CONTRIBUTING to use the environment variable CUDF_HOME
- PR #1416 Fix bug at gdf_quantile_exact and gdf_quantile_appox
- PR #1421 Fix remove creation of series multiple times during `add_column()`
- PR #1405 CSV Reader: Fix memory leaks on read_csv() failure
- PR #1328 Fix CategoricalColumn to_arrow() null mask
- PR #1433 Fix NVStrings/categories includes
- PR #1432 Update NVStrings to 0.7.* to coincide with 0.7 development
- PR #1483 Modify CSV reader to avoid cropping blank quoted characters in non-string fields
- PR #1446 Merge 1275 hotfix from master into branch-0.7
- PR #1447 Fix legacy groupby apply docstring
- PR #1451 Fix hash join estimated result size is not correct
- PR #1454 Fix local build script improperly change directory permissions
- PR #1490 Require Dask 1.1.0+ for `is_dataframe_like` test or skip otherwise.
- PR #1491 Use more specific directories & groups in CODEOWNERS
- PR #1497 Fix Thrust issue on CentOS caused by missing default constructor of host_vector elements
- PR #1498 Add missing include guard to device_atomics.cuh and separated DEVICE_ATOMICS_TEST
- PR #1506 Fix csv-write call to updated NVStrings method
- PR #1510 Added nvstrings `fillna()` function
- PR #1507 Parquet Reader: Default string data to GDF_STRING
- PR #1535 Fix doc issue to ensure correct labelling of cudf.series
- PR #1537 Fix `undefined reference` link error in HashPartitionTest
- PR #1548 Fix ci/local/build.sh README from using an incorrect image example
- PR #1551 CSV Reader: Fix integer column name indexing
- PR #1586 Fix broken `scalar_wrapper::operator==`
- PR #1591 ORC/Parquet Reader: Fix missing import for FileNotFoundError exception
- PR #1573 Parquet Reader: Fix crash due to clash with ORC reader datasource
- PR #1607 Revert change of `column.to_dense_buffer` always return by copy for performance concerns
- PR #1618 ORC reader: fix assert & data output when nrows/skiprows isn't aligned to stripe boundaries
- PR #1631 Fix failure of TYPES_TEST on some gcc-7 based systems.
- PR #1641 CSV Reader: Fix skip_blank_lines behavior with Windows line terminators (\r\n)
- PR #1648 ORC reader: fix non-deterministic output when skiprows is non-zero
- PR #1676 Fix groupby `as_index` behaviour with `MultiIndex`
- PR #1659 Fix bug caused by empty groupbys and multiindex slicing throwing exceptions
- PR #1656 Correct Groupby failure in dask when un-aggregable columns are left in dataframe.
- PR #1689 Fix groupby performance regression
- PR #1694 Add Cython as a runtime dependency since it's required in `setup.py`


# cuDF 0.6.1 (25 Mar 2019)

## Bug Fixes

- PR #1275 Fix CentOS exception in DataFrame.hash_partition from using value "returned" by a void function


# cuDF 0.6.0 (22 Mar 2019)

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
- PR #1252 Fix NVStrings dependencies for cuda 9.2 and 10.0
- PR #2037 Optimize the existing `gather` and `scatter` routines in `libcudf`

## Bug Fixes

- PR #821 Fix flake8 issues revealed by flake8 update
- PR #808 Resolved renamed `d_columns_valids` variable name
- PR #820 CSV Reader: fix the issue where reader adds additional rows when file uses \r\n as a line terminator
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
- PR #1213 Fix bug with csv reader requesting subset of columns using wrong datatype
- PR #1223 gpuCI: Fix label on rapidsai channel on gpu build scripts
- PR #1242 Add explicit Thrust exec policy to fix NVCATEGORY_TEST segfault on some platforms
- PR #1246 Fix categorical tests that failed due to bad implicit type conversion
- PR #1255 Fix overwriting conda package main label uploads
- PR #1259 Add dlpack includes to pip build


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
