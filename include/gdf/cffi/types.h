#ifndef _GDF_TYPES_H_
#define _GDF_TYPES_H_

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
	GDF_DATE32,
	GDF_DATE64,
    N_GDF_TYPES, /* additional types should go BEFORE N_GDF_TYPES */
} gdf_dtype;

typedef enum {
    GDF_SUCCESS=0,
    GDF_CUDA_ERROR,
    GDF_UNSUPPORTED_DTYPE,
    GDF_COLUMN_SIZE_MISMATCH,
    GDF_COLUMN_SIZE_TOO_BIG,
    GDF_VALIDITY_MISSING,
    GDF_VALIDITY_UNSUPPORTED,
    GDF_INVALID_API_CALL
} gdf_error;

typedef struct gdf_column_{
    void *data;
    gdf_valid_type *valid;
    gdf_size_type size;
    gdf_dtype dtype;
    gdf_size_type null_count = 0;
} gdf_column;

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

#endif