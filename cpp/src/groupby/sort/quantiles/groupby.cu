/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "groupby.hpp"

#include <copying/scatter.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include <cudf/copying.hpp>

#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <algorithm>
#include <tuple>
#include <numeric>

namespace {

template <bool nullable = true>
struct transform_row_eq_comparator {
  row_equality_comparator<nullable> cmp;
  gdf_size_type* sorted_order;

  CUDA_DEVICE_CALLABLE
  bool operator() (gdf_size_type lhs, gdf_size_type rhs) {
    return cmp(sorted_order[lhs], sorted_order[rhs]);
  }
};

} // namespace anonymous


namespace cudf {

namespace sort {

void groupby::set_key_sort_order() {
  if (_include_nulls ||
      !cudf::has_nulls(_key_table)) {  // SQL style
    gdf_context context{};
    context.flag_groupby_include_nulls = true;
    CUDF_TRY(gdf_order_by(_key_table.begin(), nullptr,
                          _key_table.num_columns(), &_key_sorted_order,
                          &context));
  } else {  // Pandas style
    auto key_cols_bitmask = row_bitmask(_key_table);

    gdf_column modified_fist_key_col = *(_key_table.get_column(0));
    modified_fist_key_col.valid =
        reinterpret_cast<gdf_valid_type*>(key_cols_bitmask.data().get());

    std::vector<gdf_column*> modified_key_cols_vect = _key_table.get_columns();
    modified_key_cols_vect[0] = &modified_fist_key_col;
    cudf::table modified_key_col_table(modified_key_cols_vect.data(),
                                      modified_key_cols_vect.size());

    gdf_context temp_ctx;
    temp_ctx.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

    CUDF_TRY(gdf_order_by(modified_key_col_table.begin(), nullptr,
                          modified_key_col_table.num_columns(),
                          &_key_sorted_order, &temp_ctx));

    CUDF_TRY(gdf_count_nonzero_mask(
        reinterpret_cast<gdf_valid_type*>(key_cols_bitmask.data().get()),
        _key_table.num_rows(),
        &_num_keys));
  }
}

void groupby::set_group_ids() {
  index_vector idx_data(_num_keys);

  auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);
  auto device_input_table = device_table::create(_key_table);
  bool nullable = device_input_table.get()->has_nulls();
  auto sorted_order = reinterpret_cast<gdf_size_type*>(_key_sorted_order.data);
  decltype(idx_data.begin()) result_end;

  if (nullable) {
    auto comp = row_equality_comparator<true>(*device_input_table, true);
    result_end = thrust::unique_copy(
      thrust::device, counting_iter, counting_iter + _num_keys,
      idx_data.begin(), transform_row_eq_comparator<true>{comp, sorted_order});
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table, true);
    result_end = thrust::unique_copy(
      thrust::device, counting_iter, counting_iter + _num_keys,
      idx_data.begin(), transform_row_eq_comparator<false>{comp, sorted_order});
  }

  gdf_size_type num_groups = thrust::distance(idx_data.begin(), result_end);
  _group_ids = index_vector(idx_data.begin(), idx_data.begin() + num_groups);
}

void groupby::set_group_labels() {
  // Get group labels for future use in segmented sorting
  _group_labels = index_vector(_num_keys);
  thrust::fill(_group_labels.begin(), _group_labels.end(), 0);
  auto group_labels_ptr = _group_labels.data().get();
  auto group_ids_ptr = _group_ids.data().get();
  thrust::for_each_n(thrust::make_counting_iterator(1),
                    _group_ids.size() - 1,
                    [=] __device__ (gdf_size_type i) { 
                      group_labels_ptr[group_ids_ptr[i]] = 1;
                    });
  thrust::inclusive_scan(thrust::device,
                        _group_labels.begin(),
                        _group_labels.end(),
                        _group_labels.begin());
}

void groupby::set_unsorted_labels() {
  _unsorted_labels = allocate_column(gdf_dtype_of<gdf_size_type>(),
                                    _key_sorted_order.size);
  cudaMemset(_unsorted_labels.valid, 0,
              gdf_num_bitmask_elements(_unsorted_labels.size));
  
  gdf_column group_labels_col{};
  gdf_column_view(&group_labels_col, _group_labels.data().get(), nullptr,
                  _group_labels.size(), gdf_dtype_of<gdf_size_type>());
  cudf::table sorted_labels{&group_labels_col};
  cudf::table unsorted_labels{&_unsorted_labels};
  cudf::detail::scatter(&sorted_labels,
                        reinterpret_cast<gdf_size_type*>(_key_sorted_order.data),
                        &unsorted_labels);
  print<gdf_size_type>(_unsorted_labels);
}


std::pair<gdf_column, rmm::device_vector<gdf_size_type> >
groupby::sort_values(gdf_column const& val_col) {
  auto idx_col = allocate_column(gdf_dtype_of<gdf_index_type>(),
                                _key_table.num_rows(),
                                false);
  auto unsorted_val_col = const_cast<gdf_column*> (&val_col);
  auto unsorted_table = cudf::table{&_unsorted_labels, unsorted_val_col};

  gdf_context context{};
  context.flag_groupby_include_nulls = _include_nulls;
  gdf_order_by(unsorted_table.begin(),
              nullptr,
              unsorted_table.num_columns(), // always 2
              &idx_col,
              const_cast<gdf_context*>(&context));

  cudf::table unsorted_val_col_table{unsorted_val_col};
  auto sorted_val_col = allocate_like(val_col, _num_keys);
  cudf::table sorted_val_col_table{&sorted_val_col};
  cudf::gather(&unsorted_val_col_table,
              reinterpret_cast<gdf_size_type*>(idx_col.data),
              &sorted_val_col_table);
  gdf_column_free(&idx_col);

  // Get number of valid values in each group
  rmm::device_vector<gdf_size_type> val_group_sizes(_group_ids.size());
  rmm::device_vector<gdf_size_type> d_bools(sorted_val_col.size);
  if ( is_nullable(sorted_val_col) ) {
    auto col_valid = reinterpret_cast<bit_mask::bit_mask_t*>(sorted_val_col.valid);

    thrust::transform(
      thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
      thrust::make_counting_iterator(sorted_val_col.size), d_bools.begin(),
      [col_valid] __device__ (gdf_size_type i) { return bit_mask::is_valid(col_valid, i); });
  } else {
    thrust::fill(d_bools.begin(), d_bools.end(), 1);
    print(d_bools, "col valids");
  }

  thrust::reduce_by_key(thrust::device,
                        _group_labels.begin(),
                        _group_labels.end(),
                        d_bools.begin(),
                        thrust::make_discard_iterator(),
                        val_group_sizes.begin());
  print(val_group_sizes, "grp size");

  return std::make_pair(sorted_val_col, val_group_sizes);
}

cudf::table groupby::unique_keys() {
  auto uniq_key_table = cudf::allocate_like(_key_table, (gdf_size_type)_group_ids.size());
  auto idx_data = reinterpret_cast<gdf_size_type*>(_key_sorted_order.data);
  auto transformed_group_ids = index_vector(_group_ids.size());

  util::cuda::scoped_stream stream;
  auto exec = rmm::exec_policy(stream)->on(stream);

  print(_group_ids, "group ids");
  thrust::transform(exec, _group_ids.begin(), _group_ids.end(),
                    transformed_group_ids.begin(),
    [=] __device__ (gdf_size_type i) { return idx_data[i]; } );
  print(transformed_group_ids, "transf group ids");
  cudaStreamSynchronize(stream);
  
  cudf::gather(&_key_table,
              transformed_group_ids.data().get(),
              &uniq_key_table);
  return uniq_key_table;
}


} // namespace sort
  
} // namespace cudf
