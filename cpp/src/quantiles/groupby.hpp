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

#pragma once

#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {

namespace detail {

template <typename T>
void print(rmm::device_vector<T> const& d_vec, std::string label = "") {
  thrust::host_vector<T> h_vec = d_vec;
  printf("%s \t", label.c_str());
  for (auto &&i : h_vec)  std::cout << i << " ";
  printf("\n");
}

template <typename T>
void print(gdf_column const& col, std::string label = "") {
  auto col_data = reinterpret_cast<T*>(col.data);
  auto d_vec = rmm::device_vector<T>(col_data, col_data+col.size);
  print(d_vec, label);
}

using null_order = groupby::sort::null_order;

struct groupby {
  using index_vector = rmm::device_vector<gdf_size_type>;

  groupby(cudf::table const& key_table, bool include_nulls = false, 
          null_order null_sort_behavior = null_order::AFTER,
          bool input_sorted = false)
  : _key_table(key_table)
  , _num_keys(key_table.num_rows())
  , _include_nulls(include_nulls)
  , _null_sort_behavior(null_sort_behavior)
  , _input_sorted(input_sorted)
  {
    _key_sorted_order = allocate_column(gdf_dtype_of<gdf_index_type>(),
                                        key_table.num_rows(),
                                        false);

    if (not _input_sorted) {
      set_key_sort_order();
    } else {
      set_key_pre_sorted_order();
    }

    if (_num_keys != 0) { // check if the number of groups is zero
      set_group_ids();
      set_group_labels();
      set_unsorted_labels();
    }
  };

  ~groupby() {
    gdf_column_free(&_key_sorted_order);
    if (num_groups() != 0) {
      gdf_column_free(&_unsorted_labels);
    }
  }

  std::pair<gdf_column, rmm::device_vector<gdf_size_type> >
  sort_values(gdf_column const& val_col);

  cudf::table unique_keys();

  gdf_size_type num_groups() { return _group_ids.size(); }

  index_vector& group_indices() { return _group_ids; }

  gdf_column         key_sorted_order() { return _key_sorted_order; }
  
  index_vector       group_labels() { return _group_labels; }

 private:
  void set_key_sort_order();

  void set_key_pre_sorted_order();

  void set_group_ids();

  void set_group_labels();

  void set_unsorted_labels();

 private:

  gdf_column         _key_sorted_order;
  gdf_column         _unsorted_labels;
  cudf::table const& _key_table;

  index_vector       _group_ids;
  index_vector       _group_labels;

  gdf_size_type      _num_keys;
  bool               _include_nulls;
  bool               _input_sorted;
  null_order         _null_sort_behavior;

};

} // namespace detail
  
} // namespace cudf
