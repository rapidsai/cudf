/* column operations */

gdf_size_type gdf_column_sizeof();

int gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                    gdf_size_type size, gdf_dtype dtype);


/* unary operators */

int gdf_sin_generic(gdf_column *input, gdf_column *output);

int gdf_sin_f32(gdf_column *input, gdf_column *output);

/* binary operators */

int gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

