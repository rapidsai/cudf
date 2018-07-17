typedef size_t gdf_size_type;
typedef gdf_size_type gdf_index_type;
typedef unsigned char gdf_valid_type;

typedef enum {
    GDF_invalid=0,
    GDF_INT8,
    GDF_INT16,
    GDF_INT32,
    GDF_INT64,
    GDF_FLOAT32,
    GDF_FLOAT64,
	GDF_DATE32,   // int32_t days since the UNIX epoch
	GDF_DATE64,   // int64_t milliseconds since the UNIX epoch
	GDF_TIMESTAMP // Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond)
} gdf_dtype;


/*
 * GDF error enum type.
 * Any changes here should be reflected in `gdf_error_get_name` as well.
 */
typedef enum {
    GDF_SUCCESS=0,
    GDF_CUDA_ERROR,
    GDF_UNSUPPORTED_DTYPE,
    GDF_COLUMN_SIZE_MISMATCH,
    GDF_COLUMN_SIZE_TOO_BIG,
    GDF_VALIDITY_MISSING,
    GDF_VALIDITY_UNSUPPORTED,
    GDF_JOIN_DTYPE_MISMATCH,
    GDF_JOIN_TOO_MANY_COLUMNS,
} gdf_error;

typedef enum {
	TIME_UNIT_NONE=0, // default (undefined)
	TIME_UNIT_s,   // second
	TIME_UNIT_ms,  // millisecond
	TIME_UNIT_us,  // microsecond
	TIME_UNIT_ns   // nanosecond
} gdf_time_unit;

typedef struct {
	gdf_time_unit time_unit;
	// here we can also hold info for decimal datatype or any other datatype that requires additional information
} gdf_dtype_extra_info;

typedef struct gdf_column_{
    void *data;
    gdf_valid_type *valid;
    gdf_size_type size;
    gdf_dtype dtype;
    gdf_dtype_extra_info dtype_info;
} gdf_column;




struct _OpaqueIpcParser;
typedef struct _OpaqueIpcParser gdf_ipc_parser_type;


struct _OpaqueRadixsortPlan;
typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;


struct _OpaqueSegmentedRadixsortPlan;
typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;


struct _OpaqueJoinResult;
typedef struct _OpaqueJoinResult gdf_join_result_type;
