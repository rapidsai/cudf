#pragma once

// TODO: Update to use fixed width types when CFFI goes away
typedef int gdf_size_type; /**< Limits the maximum size of a gdf_column to 2^31-1 */
typedef gdf_size_type gdf_index_type;
typedef unsigned char gdf_valid_type;
typedef	long	gdf_date64;
typedef	int		gdf_date32;
typedef	int		gdf_category;
typedef	long	gdf_timestamp;

/* --------------------------------------------------------------------------*/
 /**
 * @Synopsis  These enums indicate the possible data types for a gdf_column
 */
/* ----------------------------------------------------------------------------*/
typedef enum {
    GDF_invalid=0,
    GDF_INT8,
    GDF_INT16,
    GDF_INT32,
    GDF_INT64,
    GDF_FLOAT32,
    GDF_FLOAT64,
    GDF_DATE32,   	/**< int32_t days since the UNIX epoch */
    GDF_DATE64,   	/**< int64_t milliseconds since the UNIX epoch */
    GDF_TIMESTAMP,	/**< Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond) */
    GDF_CATEGORY,
    GDF_STRING,
    N_GDF_TYPES, 	/* additional types should go BEFORE N_GDF_TYPES */
} gdf_dtype;


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  These are all possible gdf error codes that can be returned from
 * a libgdf function. ANY NEW ERROR CODE MUST ALSO BE ADDED TO `gdf_error_get_name`
 * AS WELL
 */
/* ----------------------------------------------------------------------------*/
typedef enum {
    GDF_SUCCESS=0,                
    GDF_CUDA_ERROR,                    /**< Error occured in a CUDA call */ 
    GDF_UNSUPPORTED_DTYPE,             /**< The datatype of the gdf_column is unsupported */ 
    GDF_COLUMN_SIZE_MISMATCH,          /**< Two columns that should be the same size aren't the same size*/        
    GDF_COLUMN_SIZE_TOO_BIG,           /**< Size of column is larger than the max supported size */      
    GDF_DATASET_EMPTY,                 /**< Input dataset is either null or has size 0 when it shouldn't */   
    GDF_VALIDITY_MISSING,              /**< gdf_column's validity bitmask is null */  
    GDF_VALIDITY_UNSUPPORTED,          /**< The requested gdf operation does not support validity bitmask handling, and one of the input columns has the valid bits enabled */
    GDF_INVALID_API_CALL,              /**< The arguments passed into the function were invalid */   
    GDF_JOIN_DTYPE_MISMATCH,           /**< Datatype mismatch between corresponding columns in  left/right tables in the Join function */   
    GDF_JOIN_TOO_MANY_COLUMNS,         /**< Too many columns were passed in for the requested join operation*/       
    GDF_DTYPE_MISMATCH,                /**< Type mismatch between columns that should be the same type */
    GDF_UNSUPPORTED_METHOD,            /**< The method requested to perform an operation was invalid or unsupported (e.g., hash vs. sort)*/ 
    GDF_INVALID_AGGREGATOR,            /**< Invalid aggregator was specified for a groupby*/
    GDF_INVALID_HASH_FUNCTION,         /**< Invalid hash function was selected */
    GDF_PARTITION_DTYPE_MISMATCH,      /**< Datatype mismatch between columns of input/output in the hash partition function */
    GDF_HASH_TABLE_INSERT_FAILURE,     /**< Failed to insert to hash table, likely because its full */
    GDF_UNSUPPORTED_JOIN_TYPE,         /**< The type of join requested is unsupported */
    GDF_C_ERROR,                       /**< C error not related to CUDA */
    GDF_FILE_ERROR,                    /**< error processing sepcified file */      
    GDF_MEMORYMANAGER_ERROR,           /**< Memory manager error (see memory.h) */
    GDF_UNDEFINED_NVTX_COLOR,          /**< The requested color used to define an NVTX range is not defined */
    GDF_NULL_NVTX_NAME,                /**< The requested name for an NVTX range cannot be nullptr */
    GDF_TIMESTAMP_RESOLUTION_MISMATCH, /**< Resolution mismatch between two columns of GDF_TIMESTAMP */
    GDF_NOTIMPLEMENTED_ERROR,          /**< A feature is not implemented */
    N_GDF_ERRORS
} gdf_error;

typedef enum {
    GDF_HASH_MURMUR3=0, /**< Murmur3 hash function */
    GDF_HASH_IDENTITY,  /**< Identity hash function that simply returns the key to be hashed */
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
    void *data;                       /**< Pointer to the columns data */ 
    gdf_valid_type *valid;            /**< Pointer to the columns validity bit mask where the 'i'th bit indicates if the 'i'th row is NULL */
    gdf_size_type size;               /**< Number of data elements in the columns data buffer. Limited to 2^31 - 1.*/
    gdf_dtype dtype;                  /**< The datatype of the column's data */
    gdf_size_type null_count;         /**< The number of NULL values in the column's data */
    gdf_dtype_extra_info dtype_info;
    char *			col_name;			// host-side:	null terminated string
} gdf_column;

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  These enums indicate which method is to be used for an operation.
 * For example, it is used to select between the hash-based vs. sort-based implementations
 * of the Join operation.
 */
/* ----------------------------------------------------------------------------*/
typedef enum {
  GDF_SORT = 0,   /**< Indicates that the sort-based implementation of the function will be used */
  GDF_HASH,       /**< Indicates that the hash-based implementation of the function will be used */
  N_GDF_METHODS,  /* additional methods should go BEFORE N_GDF_METHODS */
} gdf_method;

typedef enum {
  GDF_QUANT_LINEAR =0,
  GDF_QUANT_LOWER,
  GDF_QUANT_HIGHER,
  GDF_QUANT_MIDPOINT,
  GDF_QUANT_NEAREST,
  N_GDF_QUANT_METHODS,
} gdf_quantile_method;


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis These enums indicate the supported aggregation operations that can be
 * performed on a set of aggregation columns as part of a GroupBy operation
 */
/* ----------------------------------------------------------------------------*/
typedef enum {
  GDF_SUM = 0,        /**< Computes the sum of all values in the aggregation column*/
  GDF_MIN,            /**< Computes minimum value in the aggregation column */
  GDF_MAX,            /**< Computes maximum value in the aggregation column */
  GDF_AVG,            /**< Computes arithmetic mean of all values in the aggregation column */
  GDF_COUNT,          /**< Computes histogram of the occurance of each key in the GroupBy Columns */
  GDF_COUNT_DISTINCT, /**< Counts the number of distinct keys in the GroupBy columns */
  N_GDF_AGG_OPS,      /**< The total number of aggregation operations. ALL NEW OPERATIONS SHOULD BE ADDED ABOVE THIS LINE*/
} gdf_agg_op;


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Colors for use with NVTX ranges.
 *
 * These enumerations are the available pre-defined colors for use with
 * user-defined NVTX ranges.
 */
/* ----------------------------------------------------------------------------*/
typedef enum {
  GDF_GREEN = 0, 
  GDF_BLUE,
  GDF_YELLOW,
  GDF_PURPLE,
  GDF_CYAN,
  GDF_RED,
  GDF_WHITE,
  GDF_DARK_GREEN,
  GDF_ORANGE,
  GDF_NUM_COLORS, /** Add new colors above this line */
} gdf_color;

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This struct holds various information about how an operation should be 
 * performed as well as additional information about the input data.
 */
/* ----------------------------------------------------------------------------*/
typedef struct gdf_context_{
  int flag_sorted;              /**< Indicates if the input data is sorted. 0 = No, 1 = yes */
  gdf_method flag_method;       /**< The method to be used for the operation (e.g., sort vs hash) */
  int flag_distinct;            /**< for COUNT: DISTINCT = 1, else = 0 */
  int flag_sort_result;         /**< When method is GDF_HASH, 0 = result is not sorted, 1 = result is sorted */
  int flag_sort_inplace;        /**< 0 = No sort in place allowed, 1 = else */
} gdf_context;

struct _OpaqueIpcParser;
typedef struct _OpaqueIpcParser gdf_ipc_parser_type;


struct _OpaqueRadixsortPlan;
typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;


struct _OpaqueSegmentedRadixsortPlan;
typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;




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
