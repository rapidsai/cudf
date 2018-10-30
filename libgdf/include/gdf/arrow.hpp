#include "cffi/types.h"
#include <arrow/api.h>

void* arrow_to_gdf(arrow::PrimitiveArray *array, gdf_column* result);
std::shared_ptr<arrow::PrimitiveArray> gdf_to_arrow(gdf_column* column);
typedef arrow::Type::type arrow_type;
