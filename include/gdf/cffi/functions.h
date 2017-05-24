/* column operations */

gdf_size_type gdf_column_sizeof();

gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype);

/* error handling */

const char * gdf_error_get_name(gdf_error errcode);

/* unary operators */

gdf_error gdf_sin_generic(gdf_column *input, gdf_column *output);

gdf_error gdf_sin_f32(gdf_column *input, gdf_column *output);

/* binary operators */

gdf_error gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
