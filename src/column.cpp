#include <gdf/gdf.h>


gdf_size_type gdf_column_sizeof() {
    return sizeof(gdf_column);
}

gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                    gdf_size_type size, gdf_dtype dtype) {
    column->data = data;
    column->valid = valid;
    column->size = size;
    column->dtype = dtype;
    return GDF_SUCCESS;
}


gdf_error gdf_column_view_augmented(gdf_column *column, void *data, gdf_valid_type *valid,
                    gdf_size_type size, gdf_dtype dtype, gdf_size_type null_count) {
    column->data = data;
    column->valid = valid;
    column->size = size;
    column->dtype = dtype;
    column->null_count = null_count;
    return GDF_SUCCESS;
}
