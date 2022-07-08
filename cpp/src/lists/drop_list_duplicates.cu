/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf::lists {
namespace detail {

namespace {

/**
 * @brief Common execution code called by all public `drop_list_duplicates` APIs.
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates_common(
  lists_column_view const& keys,
  std::optional<lists_column_view> const& values,
  null_equality nulls_equal,
  nan_equality nans_equal,
  duplicate_keep_option keep_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!values || (keys.size() == values.value().size()),
               "Keys and values columns must have the same size.");

  if (keys.is_empty()) {
    return std::pair{cudf::empty_like(keys.parent()),
                     values ? cudf::empty_like(values.value().parent()) : nullptr};
  }

  auto const keys_child   = keys.get_sliced_child(stream);
  auto const values_child = values ? values.value().get_sliced_child(stream) : column_view{};
  if (values) {
    CUDF_EXPECTS(keys_child.size() == values_child.size(),
                 "Keys and values children columns must have the same size.");
  }

  auto const labels = generate_labels(keys, keys_child.size(), stream);

  auto const input_table = values ? table_view{{labels->view(), keys_child, values_child}}
                                  : table_view{{labels->view(), keys_child}};

  auto distinct_columns = cudf::detail::stable_distinct(input_table,
                                                        std::vector<size_type>{0, 1},  // keys
                                                        keep_option,
                                                        nulls_equal,
                                                        nans_equal,
                                                        stream,
                                                        mr)
                            ->release();

  auto out_offsets = reconstruct_offsets(distinct_columns.front()->view(), keys.size(), stream, mr);
  auto out_values =
    values ? make_lists_column(keys.size(),
                               std::make_unique<column>(out_offsets->view(), stream, mr),
                               std::move(distinct_columns.back()),
                               values.value().null_count(),
                               cudf::detail::copy_bitmask(values.value().parent(), stream, mr),
                               stream,
                               mr)
           : nullptr;
  auto out_keys = make_lists_column(keys.size(),
                                    std::move(out_offsets),
                                    std::move(distinct_columns[1]),
                                    keys.null_count(),
                                    cudf::detail::copy_bitmask(keys.parent(), stream, mr),
                                    stream,
                                    mr);

  return std::pair{std::move(out_keys), std::move(out_values)};
}

}  // anonymous namespace

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates(
  lists_column_view const& keys,
  lists_column_view const& values,
  null_equality nulls_equal,
  nan_equality nans_equal,
  duplicate_keep_option keep_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return drop_list_duplicates_common(keys,
                                     std::optional<lists_column_view>(values),
                                     nulls_equal,
                                     nans_equal,
                                     keep_option,
                                     stream,
                                     mr);
}

std::unique_ptr<column> drop_list_duplicates(lists_column_view const& input,
                                             null_equality nulls_equal,
                                             nan_equality nans_equal,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return drop_list_duplicates_common(input,
                                     std::nullopt,
                                     nulls_equal,
                                     nans_equal,
                                     duplicate_keep_option::KEEP_FIRST,
                                     stream,
                                     mr)
    .first;
}

}  // namespace detail

/**
 * @copydoc cudf::lists::drop_list_duplicates(lists_column_view const&,
 *                                            lists_column_view const&,
 *                                            duplicate_keep_option,
 *                                            null_equality,
 *                                            nan_equality,
 *                                            rmm::mr::device_memory_resource*)
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates(
  lists_column_view const& keys,
  lists_column_view const& values,
  duplicate_keep_option keep_option,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_list_duplicates(
    keys, values, nulls_equal, nans_equal, keep_option, cudf::default_stream_value, mr);
}

/**
 * @copydoc cudf::lists::drop_list_duplicates(lists_column_view const&,
 *                                            null_equality,
 *                                            nan_equality,
 *                                            rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& input,
                                             null_equality nulls_equal,
                                             nan_equality nans_equal,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_list_duplicates(
    input, nulls_equal, nans_equal, cudf::default_stream_value, mr);
}

}  // namespace cudf::lists
