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
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/gather.h>

namespace cudf::groupby::detail {

// Fixed type for counting frequencies in historam.
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
  return make_lists_column(
    0, make_empty_column(type_to_id<size_type>()), std::move(structs), 0, {});
}

std::unique_ptr<column> build_histogram(column_view const& values,
                                        cudf::device_span<size_type const> group_labels,
                                        std::optional<column_view> const& partial_counts,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(num_groups >= 0, "Number of groups cannot be negative.");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels.");

  if (num_groups == 0) { return make_empty_histogram(values); }

  // Attach group labels to the input values.
  auto const labels_cv      = column_view{data_type{type_to_id<size_type>()},
                                     static_cast<size_type>(group_labels.size()),
                                     group_labels.data(),
                                     nullptr,
                                     0};
  auto const labeled_values = table_view{{labels_cv, values}};

  // Build histogram for the labeled values.
  auto [distinct_indices, distinct_counts] = cudf::reduction::detail::table_histogram(
    labeled_values, partial_counts, histogram_count_dtype, stream, mr);

  // Gather the distinct rows for output histogram.
  auto out_table = cudf::detail::gather(labeled_values,
                                        distinct_indices,
                                        out_of_bounds_policy::DONT_CHECK,
                                        cudf::detail::negative_index_policy::NOT_ALLOWED,
                                        stream,
                                        mr);

  // Build offsets for the output lists column.
  // Each list will be a histogram corresponding to each value group.
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

std::unique_ptr<column> group_histogram(column_view const& values,
                                        cudf::device_span<size_type const> group_labels,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return build_histogram(values, group_labels, std::nullopt, num_groups, stream, mr);
}

std::unique_ptr<column> group_merge_histogram(column_view const& values,
                                              cudf::device_span<size_type const> group_offsets,
                                              size_type num_groups,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // The input must be a lists column without nulls.
  CUDF_EXPECTS(!values.has_nulls(), "The input column must not have nulls.");
  CUDF_EXPECTS(values.type().id() == type_id::LIST,
               "The input of MERGE_HISTOGRAM aggregation must be a lists column.");

  // Child of the input lists column must be a structs column without nulls,
  // and its second child is a count columns of integer type having no nulls.
  auto const lists_cv     = lists_column_view{values};
  auto const histogram_cv = lists_cv.get_sliced_child(stream);
  CUDF_EXPECTS(!histogram_cv.has_nulls(), "Child of the input lists column must not have nulls.");
  CUDF_EXPECTS(histogram_cv.type().id() == type_id::STRUCT && histogram_cv.num_children() == 2,
               "The input column has invalid histograms structure.");
  CUDF_EXPECTS(
    cudf::is_integral(histogram_cv.child(1).type()) && !histogram_cv.child(1).has_nulls(),
    "The input column has invalid histograms structure.");

  if (num_groups == 0) { return empty_like(values); }

  // Firstly concatenate the histograms corresponding to the same key values.
  // That is equivalent to creating a new lists column (view) from the input lists column
  // with new offsets as below.
  auto new_offsets = rmm::device_uvector<size_type>(num_groups + 1, stream);
  thrust::gather(rmm::exec_policy(stream),
                 group_offsets.begin(),
                 group_offsets.end(),
                 lists_cv.offsets_begin(),
                 new_offsets.begin());

  auto key_labels = rmm::device_uvector<size_type>(histogram_cv.size(), stream);
  cudf::detail::label_segments(
    new_offsets.begin(), new_offsets.end(), key_labels.begin(), key_labels.end(), stream);

  // The input values column is already in histogram format (i.e., column of Struct<value, count>).
  auto const structs_cv   = structs_column_view{histogram_cv};
  auto const input_values = structs_cv.get_sliced_child(0, stream);
  auto const input_counts = structs_cv.get_sliced_child(1, stream);

  return build_histogram(input_values, key_labels, input_counts, num_groups, stream, mr);
}

}  // namespace cudf::groupby::detail
