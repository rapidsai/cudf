#include "cudf.h"
#include "utilities/error_utils.h"
#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

typedef unsigned char gdf_interchange_valid_type;
typedef int32_t
    gdf_interchange_size_type;  // Limits the maximum size of a
                                 // gdf_interchange_column to 2^31-1
typedef enum {
  invalid = 0,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
} gdf_interchange_dtype;

// We define a simple interchange format loosely based on the internal column
// structure of gdf (cuda data frame). gdf is responsible for passing this
// exact structure to the xgboost C API for DMatrix construction. The decoupled
// interchange format means xgboost does not depend on the specific internal
// structure of gdf which may change.

typedef struct gdf_interchange_column_ {
  void* data;  // Pointer to the columns data
  gdf_interchange_valid_type*
      valid;  // Pointer to the columns validity bit mask where the
              // 'i'th bit indicates if the 'i'th row is NULL
  gdf_interchange_size_type size;  // Number of data elements in the columns
                                    // data buffer. Limited to 2^31 - 1.
  gdf_interchange_dtype dtype;     // The datatype of the column's data
  int32_t null_count;  // The number of NULL values in the column's data
  char* col_name;      // host-side: null terminated string
  gdf_interchange_column_() {
    static_assert(sizeof(gdf_interchange_column_) == 40,
                  "If this static assert fails, the compiler is not supported "
                  "- please file an issue");
  }
} gdf_interchange_column;