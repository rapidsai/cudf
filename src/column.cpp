#include <gdf/gdf.h>


gdf_size_type gdf_column_sizeof() {
    return sizeof(gdf_column);
}

int gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                    gdf_size_type size, gdf_dtype dtype) {
    column->data = data;
    column->valid = valid;
    column->size = size;
    column->dtype = dtype;
    return 0;
}

