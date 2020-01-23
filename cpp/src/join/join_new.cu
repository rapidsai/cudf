/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/join_new.hpp>

namespace cudf {

namespace join {

namespace detail {

  // TODO:
  //  nested_join_indices      - my function/kernel
  //  sort_merge_join_indices  - Kumar
  //  hash_join_indices        - rework current hash mechanism
  //  filter_join_indices      - new function/kernel
  //
  struct nested_loop_join {
    VectorPair join_indices(table_view const& left,
                            table_view const& right,
                            std::vector<join_operation> const& primary_join_ops,
                            std::vector<join_operation> const& secondary_join_ops,
                            cudaStream_t stream) {

      return nested_join_indices(left, right, primary_join_ops, stream);
    }
  };

  struct sort_merge_join {
    VectorPair operator()(table_view const& left,
                          table_view const& right,
                          std::vector<join_operation> const& primary_join_ops,
                          std::vector<join_operation> const& secondary_join_ops,
                          cudaStream_t stream) {

      auto joined_indices = sort_merge_join_indices(left, right, primary_join_ops, stream);

      if (secondary_join_ops.empty()) {
        return joined_indices;
      }

      return filter_join_indices(left, right, joined_indices, secondary_join_ops, stream);
    }
  };

  struct hash_join {
    VectorPair operator()(table_view const& left,
                          table_view const& right,
                          std::vector<join_operation> const& primary_join_ops,
                          std::vector<join_operation> const& secondary_join_ops,
                          cudaStream_t stream) {

      auto joined_indices = hash_join_indices(left, right, primary_join_ops, stream);

      if (secondary_join_ops.empty()) {
        return joined_indices;
      }

      return filter_join_indices(left, right, joined_indices, secondary_join_ops, stream);
    }
  };
    
  bool is_trivial_join(table_view const& left,
                       table_view const& right,
                       std::vector<join_operation> const& primary_join_ops,
                       std::vector<join_operation> const& secondary_join_ops,
                       join_kind JoinKind) {

    // If there is nothing to join, then send empty table with all columns
    if (primary_join_ops.empty() && secondary_join_ops.empty()) {
      return true;
    }

    // If the inputs are empty, immediately return
    if ((0 == left.num_rows()) && (0 == right.num_rows())) {
      return true;
    }

    // If left join and the left table is empty, return immediately
    if ((join_kind::LEFT_JOIN == JoinKind) && (0 == left.num_rows())) {
      return true;
    }

    // If Inner Join and either table is empty, return immediately
    if ((join_kind::INNER_JOIN == JoinKind) &&
        ((0 == left.num_rows()) || (0 == right.num_rows()))) {
      return true;
    }

    return false;
  }

  template <typename join_indices, join_kind JoinKind>
  std::unique_ptr<table> join(cudf::table_view const& left,
                              cudf::table_view const& right,
                              std::vector<join_operation> const& primary_join_ops,
                              std::vector<join_operation> const& secondary_join_ops,
                              std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                              join_indices join_indices_impl,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream = 0) {

    CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
    CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");

    if (is_trivial_join(left, right, primary_join_ops, secondary_join_ops, JoinKind)) {
      return get_empty_joined_table(left, right, columns_in_common);
    }

    //
    //  Call nested_loop on join_ops to get joined_indices
    //
    auto joined_indices = join_indices_impl(left, right, primary_join_ops, secondary_join_ops, stream);

    if (joined_indices.size() == 0) {
      return get_empty_joined_table(left, right, columns_in_common);
    } else {
      return construct_join_output_df<JoinKind>(left, right, joined_indices, columns_in_common, mr, stream);
    }
  }
} //namespace detail

namespace nested_loop {
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<join_operation> const& join_ops,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {

    return detail::join<detail::nested_loop_join, join_kind::INNER_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::nested_loop_join, join_kind::LEFT_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::nested_loop_join, join_kind::FULL_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }
} //namespace nested_loop

namespace sort_merge {
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<join_operation> const& primary_join_ops,
                                    std::vector<join_operation> const& secondary_join_ops,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::sort_merge_join, join_kind::INNER_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::sort_merge_join, join_kind::LEFT_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::sort_merge_join, join_kind::FULL_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }
} //namespace sort_merge
  
namespace hash {
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<join_operation> const& primary_join_ops,
                                    std::vector<join_operation> const& secondary_join_ops,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::hash_join, join_kind::INNER_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::hash_join{}, mr);
  }

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::hash_join, join_kind::LEFT_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::hash_join{}, mr);
  }

  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return detail::join<detail::hash_join, join_kind::FULL_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::hash_join{}, mr);
  }
} //namespace hash
  
} //namespace join

} //namespace cudf
