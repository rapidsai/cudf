/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <lists/utilities.hpp>
#include <reductions/histogram_helpers.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

namespace cudf::groupby::detail {

// Fixed type for counting frequencies in historam.
// This is to avoid using `target_type_t` which requires type_dispatcher.
constexpr auto histogram_count_dtype = data_type{type_to_id<int64_t>()};

namespace {
auto make_empty_histogram(column_view const& values)
{
  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(empty_like(values));
  struct_children.emplace_back(make_numeric_column(histogram_count_dtype, 0));
  auto structs = std::make_unique<column>(data_type{type_id::STRUCT},
                                          0,
                                          rmm::device_buffer{},
                                          rmm::device_buffer{},
                                          0,
                                          std::move(struct_children));

  std::vector<std::unique_ptr<column>> lists_children;
  lists_children.emplace_back(make_numeric_column(data_type{type_to_id<size_type>()}, 0));
  lists_children.emplace_back(std::move(structs));
  return std::make_unique<column>(cudf::data_type{type_id::LIST},
                                  0,
                                  rmm::device_buffer{},
                                  rmm::device_buffer{},
                                  0,
                                  std::move(lists_children));
}

std::unique_ptr<column> histogram(column_view const& input,
                                  cudf::device_span<size_type const> group_labels,
                                  std::optional<column_view> const& partial_counts,
                                  size_type num_groups,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(input.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  if (num_groups == 0) { return make_empty_histogram(input); }

  auto const labels_cv      = column_view{data_type{type_to_id<size_type>()},
                                     static_cast<size_type>(group_labels.size()),
                                     group_labels.data(),
                                     nullptr,
                                     0};
  auto const labeled_values = table_view{{labels_cv, input}};

  auto [distinct_indices, distinct_counts] = cudf::reduction::detail::table_histogram(
    labeled_values, partial_counts, histogram_count_dtype, stream, mr);
  auto out_table = cudf::detail::gather(labeled_values,
                                        distinct_indices,
                                        out_of_bounds_policy::DONT_CHECK,
                                        cudf::detail::negative_index_policy::NOT_ALLOWED,
                                        stream,
                                        mr);

  auto out_offsets = cudf::lists::detail::reconstruct_offsets(
    out_table->get_column(0).view(), num_groups, stream, mr);

  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(std::move(out_table->release().back()));
  struct_children.emplace_back(std::move(distinct_counts));
  auto out_structs = make_structs_column(
    static_cast<size_type>(distinct_indices.size()), std::move(struct_children), 0, {}, stream, mr);

  return make_lists_column(
    num_groups, std::move(out_offsets), std::move(out_structs), 0, {}, stream, mr);
}

}  // namespace

std::unique_ptr<column> group_histogram(column_view const& input,
                                        cudf::device_span<size_type const> group_labels,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return histogram(input, group_labels, std::nullopt, num_groups, stream, mr);
}

std::unique_ptr<column> group_merge_histogram(column_view const& input,
                                              cudf::device_span<size_type const> group_labels,
                                              size_type num_groups,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!input.has_nulls(), "The input column must not have nulls.");
  CUDF_EXPECTS(
    input.type().id() == type_id::STRUCT && input.num_children() == 2,
    "The input of merge_histogram aggregation must be a struct column having two children.");
  CUDF_EXPECTS(cudf::is_integral(input.child(1).type()) && !input.child(1).has_nulls(),
               "The second child of the input column must be ingegral type and has no nulls.");

  if (num_groups == 0) { return empty_like(input); }

  auto const structs_cv   = structs_column_view{input};
  auto const input_values = structs_cv.get_sliced_child(0, stream);
  auto const input_counts = structs_cv.get_sliced_child(1, stream);

  return histogram(input_values, group_labels, input_counts, num_groups, stream, mr);
}

}  // namespace cudf::groupby::detail
