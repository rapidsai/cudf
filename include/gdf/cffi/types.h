
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
    GDF_TIMESTAMP,// Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond)
    N_GDF_TYPES, /* additional types should go BEFORE N_GDF_TYPES */
} gdf_dtype;

typedef enum {
    GDF_SUCCESS=0,
    GDF_CUDA_ERROR,
    GDF_UNSUPPORTED_DTYPE,
    GDF_COLUMN_SIZE_MISMATCH,
    GDF_COLUMN_SIZE_TOO_BIG,
    GDF_DATASET_EMPTY,
    GDF_VALIDITY_MISSING,
    GDF_VALIDITY_UNSUPPORTED,
    GDF_INVALID_API_CALL,
    GDF_JOIN_DTYPE_MISMATCH,
    GDF_JOIN_TOO_MANY_COLUMNS,
    GDF_UNSUPPORTED_METHOD,
} gdf_error;

typedef enum {
    GDF_HASH_MURMUR3=0,
} gdf_hash_func;

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
    gdf_size_type null_count;
    gdf_dtype_extra_info dtype_info;
} gdf_column;

typedef enum {
  GDF_SORT = 0,
  GDF_HASH,
  N_GDF_METHODS,  /* additional methods should go BEFORE N_GDF_METHODS */
} gdf_method;

typedef enum {
  GDF_SUM = 0,
  GDF_MIN,
  GDF_MAX,
  GDF_AVG,
  GDF_COUNT,
  GDF_COUNT_DISTINCT,
  N_GDF_AGG_OPS, /* additional aggregation ops should go BEFORE N_GDF_... */
} gdf_agg_op;

/* additonal flags */
typedef struct gdf_context_{
  int flag_sorted;        /* 0 = No, 1 = yes */
  gdf_method flag_method; /* what method is used */
  int flag_distinct;      /* for COUNT: DISTINCT = 1, else = 0 */
} gdf_context;

struct _OpaqueIpcParser;
typedef struct _OpaqueIpcParser gdf_ipc_parser_type;


struct _OpaqueRadixsortPlan;
typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;


struct _OpaqueSegmentedRadixsortPlan;
typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;


struct _OpaqueJoinResult;
typedef struct _OpaqueJoinResult gdf_join_result_type;


typedef enum{
	GDF_ORDER_ASC,
	GDF_ORDER_DESC
} order_by_type;

typedef enum{
	GDF_EQUALS,
	GDF_NOT_EQUALS,
	GDF_LESS_THAN,
	GDF_LESS_THAN_OR_EQUALS,
	GDF_GREATER_THAN,
	GDF_GREATER_THAN_OR_EQUALS
} gdf_comparison_operator;

typedef enum{
	GDF_WINDOW_RANGE,
	GDF_WINDOW_ROW
} window_function_type;

typedef enum{
	GDF_WINDOW_AVG,
	GDF_WINDOW_SUM,
	GDF_WINDOW_MAX,
	GDF_WINDOW_MIN,
	GDF_WINDOW_COUNT,
	GDF_WINDOW_STDDEV,
	GDF_WINDOW_VAR //variance
} window_reduction_type;
