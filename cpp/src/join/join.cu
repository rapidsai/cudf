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
namespace cudf {

template <JoinType join_type, typename index_type>
std::unique_ptr<cudf::table> join_call_compute_df(
                         cudf::table_view const& left,
                         cudf::table_view const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudaStream_t stream=0,
                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");
  CUDF_EXPECTS(std::none_of(std::cbegin(left), std::cend(left), [](auto col) { return col->dtype == GDF_invalid; }), "Unsupported left column dtype");
  CUDF_EXPECTS(std::none_of(std::cbegin(right), std::cend(right), [](auto col) { return col->dtype == GDF_invalid; }), "Unsupported right column dtype");

  std::vector<cudf::size_type> right_columns_in_common (columns_in_common.size());
  for (unsigned int i = 0; i < columns_in_common.size(); ++i) {
    right_columns_in_common [i] = columns_in_common[i].second;
  }
  std::vector <cudf::size_type> right_non_common_indices = non_common_column_indices(right.num_columns(),
                                                                         right_columns_in_common);;
}

std::unique_ptr<cudf::table> inner_join(
                             cudf::table_view const& left,
                             cudf::table_view const& right,
                             std::vector<cudf::size_type> const& left_on,
                             std::vector<cudf::size_type> const& right_on,
                             std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                             cudaStream_t stream=0,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return join_call_compute_df<JoinType::INNER_JOIN, output_index_type>(
        left,
        right,
        left_on,
        right_on,
        columns_in_common,
        stream,
        mr);
}

std::unique_ptr<cudf::table> left_join(
                             cudf::table_view const& left,
                             cudf::table_view const& right,
                             std::vector<cudf::size_type> const& left_on,
                             std::vector<cudf::size_type> const& right_on,
                             std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                             cudaStream_t stream=0,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return join_call_compute_df<JoinType::LEFT_JOIN, output_index_type>(
           left,
           right,
           left_on,
           right_on,
           columns_in_common,
           stream,
           mr);
}

std::unique_ptr<cudf::table> full_join(
                             cudf::table_view const& left,
                             cudf::table_view const& right,
                             std::vector<cudf::size_type> const& left_on,
                             std::vector<cudf::size_type> const& right_on,
                             std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                             cudaStream_t stream=0,
                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
    return join_call_compute_df<JoinType::FULL_JOIN, output_index_type>(
           left,
           right,
           left_on,
           right_on,
           columns_in_common,
           stream,
           mr);
}

}
