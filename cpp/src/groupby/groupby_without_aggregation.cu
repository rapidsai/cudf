#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <utilities/nvtx/nvtx_utils.h>
#include <utilities/error_utils.hpp>
#include <string/nvcategory_util.hpp>
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>

#include <cassert>
#include <thrust/fill.h>
#include <algorithm>
#include <tuple>

rmm::device_vector<gdf_index_type>
gdf_unique_indices(cudf::table const& input_table, gdf_context const& context)
{
  gdf_size_type ncols = input_table.num_columns();
  gdf_size_type nrows = input_table.num_rows();

  rmm::device_vector<void*> d_cols(ncols);
  rmm::device_vector<int> d_types(ncols, 0);
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();

  gdf_index_type* result_end;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  rmm::device_vector<gdf_index_type> unique_indices(nrows);

  auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);
  auto device_input_table = device_table::create(input_table);
  bool nullable = device_input_table.get()->has_nulls();
  if (nullable){
    auto comp = row_equality_comparator<true>(*device_input_table, true);
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows,
                              unique_indices.data().get(),
                              comp);
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table, true);
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows,
                              unique_indices.data().get(),
                              comp);
  }

  unique_indices.resize(thrust::distance(unique_indices.data().get(), result_end));

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return unique_indices;
}

std::pair<cudf::table, rmm::device_vector<gdf_index_type>> 
gdf_group_by_without_aggregations(cudf::table const& input_table,
                                  gdf_size_type num_key_cols,
                                  gdf_index_type const * key_col_indices,
                                  gdf_context* context)
{
  CUDF_EXPECTS(nullptr != key_col_indices, "key_col_indices is null");
  CUDF_EXPECTS(0 < num_key_cols, "number of key colums should be greater than zero");

  if (0 == input_table.num_rows()) {
    return std::make_pair(cudf::table(), rmm::device_vector<gdf_index_type>());
  }

  gdf_size_type nrows = input_table.num_rows();
  
  // Allocate output columns
  cudf::table destination_table(nrows, cudf::column_dtypes(input_table), true);

  std::vector<gdf_column*> key_cols_vect(num_key_cols);
  std::transform(key_col_indices, key_col_indices+num_key_cols, key_cols_vect.begin(),
                  [&input_table] (gdf_index_type const index) { return const_cast<gdf_column*>(input_table.get_column(index)); });
  cudf::table key_col_table(key_cols_vect.data(), key_cols_vect.size());

  rmm::device_vector<gdf_size_type> sorted_indices(nrows);
  gdf_column sorted_indices_col{};
  CUDF_TRY(gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()),
                          nullptr, nrows, GDF_INT32));

  if (context->flag_groupby_include_nulls || !cudf::has_nulls(key_col_table)){  // SQL style
    CUDF_TRY(gdf_order_by(key_col_table.begin(),
                          nullptr,
                          key_col_table.num_columns(),
                          &sorted_indices_col,
                          context));
  } else {  // Pandas style

    // Pandas style ignores groups that have nulls in their keys, so we want to filter them out.
    // We will create a bitmask (key_cols_bitmask) that represents if there is any null in any of they key columns.
    // We create a modified set of key columns (modified_key_col_table), where the first key column will take this bitmask (key_cols_bitmask)
    // Then if we set flag_null_sort_behavior = GDF_NULL_AS_LARGEST, then when we sort by the key columns, 
    // then all the rows where any of the key columns contained a null, these will be at the end of the sorted set.
    // Then we can figure out how many of those rows contained any nulls and adjust the size of our sorted data set 
    // to ignore the rows where there were any nulls in the key columns
    
    auto key_cols_bitmask = row_bitmask(key_col_table);

    gdf_column modified_fist_key_col{}; 
    modified_fist_key_col.data = key_cols_vect[0]->data;
    modified_fist_key_col.size = key_cols_vect[0]->size;
    modified_fist_key_col.dtype = key_cols_vect[0]->dtype;
    modified_fist_key_col.null_count = key_cols_vect[0]->null_count;
    modified_fist_key_col.valid = reinterpret_cast<gdf_valid_type*>(key_cols_bitmask.data().get());

    std::vector<gdf_column*> modified_key_cols_vect = key_cols_vect;
    modified_key_cols_vect[0] = &modified_fist_key_col;
    cudf::table modified_key_col_table(modified_key_cols_vect.data(), modified_key_cols_vect.size());

    gdf_context temp_ctx;
    temp_ctx.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

    CUDF_TRY(gdf_order_by(modified_key_col_table.begin(),
                          nullptr,
                          modified_key_col_table.num_columns(),
                          &sorted_indices_col,
                          &temp_ctx));
    
    int valid_count;
    CUDF_TRY(gdf_count_nonzero_mask(reinterpret_cast<gdf_valid_type*>(key_cols_bitmask.data().get()),
                                    nrows, &valid_count));

    std::for_each(destination_table.begin(), destination_table.end(), 
                  [valid_count](gdf_column * col){ col->size = valid_count; });
  }

  // run gather operation to establish new order
  cudf::gather(&input_table, sorted_indices.data().get(), &destination_table);

  std::vector<gdf_column*> key_cols_vect_out(num_key_cols);
  std::transform(key_col_indices, key_col_indices+num_key_cols, key_cols_vect_out.begin(),
                  [&destination_table] (gdf_index_type const index) { return destination_table.get_column(index); });
  cudf::table key_col_sorted_table(key_cols_vect_out.data(), key_cols_vect_out.size());
  
  return std::make_pair(destination_table, gdf_unique_indices(key_col_sorted_table, *context));
}
