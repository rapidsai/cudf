/* column operations */

gdf_size_type gdf_column_sizeof();

gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype);

/* error handling */

const char * gdf_error_get_name(gdf_error errcode);


/* ipc */

ipc_parser_type* gdf_ipc_parser_open(const char *device_bytes);
void gdf_ipc_parser_close(ipc_parser_type *handle);
int gdf_ipc_parser_failed(ipc_parser_type *handle);
const char* gdf_ipc_parser_to_json(ipc_parser_type *handle);
const char* gdf_ipc_parser_get_error(ipc_parser_type *handle);
const void* gdf_ipc_parser_get_data(ipc_parser_type *handle);
int64_t gdf_ipc_parser_get_data_offset(ipc_parser_type *handle);

/* unary operators */

/* trig */

gdf_error gdf_sin_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_sin_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_sin_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_cos_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_cos_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cos_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_tan_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_tan_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_tan_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_asin_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_asin_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_asin_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_acos_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_acos_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_acos_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_atan_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_atan_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_atan_f64(gdf_column *input, gdf_column *output);

/* exponential */

gdf_error gdf_exp_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_exp_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_exp_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_log_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_log_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_log_f64(gdf_column *input, gdf_column *output);

/* power */

gdf_error gdf_sqrt_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_sqrt_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_sqrt_f64(gdf_column *input, gdf_column *output);


/* rounding */

gdf_error gdf_ceil_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_ceil_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_ceil_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_floor_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_floor_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_floor_f64(gdf_column *input, gdf_column *output);

/* casting */

gdf_error gdf_cast_generic_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_f32(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_i8(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_i32(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_i64(gdf_column *input, gdf_column *output);

/* binary operators */

/* artih */

gdf_error gdf_add_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_sub_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_mul_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_floordiv_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_div_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_div_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_div_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

/* logical */

gdf_error gdf_gt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_ge_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_lt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_le_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_eq_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_ne_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
