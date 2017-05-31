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
} gdf_dtype;

typedef enum {
    GDF_SUCCESS=0,
    GDF_CUDA_ERROR,
    GDF_UNSUPPORTED_DTYPE,
    GDF_COLUMN_SIZE_MISMATCH,
} gdf_error;

typedef struct gdf_column_{
    void *data;
    gdf_valid_type *valid;
    gdf_size_type size;
    gdf_dtype dtype;
} gdf_column;

struct _OpaqueIpcParser;
typedef struct _OpaqueIpcParser ipc_parser_type;
