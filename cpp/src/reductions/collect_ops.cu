/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace reduction {

namespace {

/**
 * @brief Check if we need to handle nulls in the input column.
 *
 * @param input The input column
 * @param null_handling The null handling policy
 * @return A boolean value indicating if we need to handle nulls
 */
bool need_handle_nulls(column_view const& input, null_policy null_handling)
{
  return null_handling == null_policy::EXCLUDE && input.has_nulls();
}

}  // namespace

std::unique_ptr<scalar> collect_list(column_view const& col,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  if (need_handle_nulls(col, null_handling)) {
    auto d_view             = column_device_view::create(col, stream);
    auto filter             = detail::validity_accessor(*d_view);
    auto null_purged_table  = detail::copy_if(table_view{{col}}, filter, stream, mr);
    column* null_purged_col = null_purged_table->release().front().release();
    null_purged_col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
    return std::make_unique<list_scalar>(std::move(*null_purged_col), true, stream, mr);
  } else {
    return make_list_scalar(col, stream, mr);
  }
}

std::unique_ptr<scalar> merge_lists(lists_column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto flatten_col = col.get_sliced_child(stream);
  return make_list_scalar(flatten_col, stream, mr);
}

std::unique_ptr<scalar> collect_set(column_view const& col,
                                    null_policy null_handling,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  // `input_as_collect_list` is the result of the input column that has been processed to obey
  // the given null handling behavior.
  [[maybe_unused]] auto const [input_as_collect_list, unused_scalar] = [&] {
    if (need_handle_nulls(col, null_handling)) {
      // Only call `collect_list` when we need to handle nulls.
      auto scalar = collect_list(col, null_handling, stream, mr);
      return std::pair(static_cast<list_scalar*>(scalar.get())->view(), std::move(scalar));
    }

    return std::pair(col, std::unique_ptr<scalar>(nullptr));
  }();

  auto distinct_table = detail::distinct(table_view{{input_as_collect_list}},
                                         std::vector<size_type>{0},
                                         duplicate_keep_option::KEEP_ANY,
                                         nulls_equal,
                                         nans_equal,
                                         stream,
                                         mr);

  return std::make_unique<list_scalar>(std::move(distinct_table->get_column(0)), true, stream, mr);
}

std::unique_ptr<scalar> merge_sets(lists_column_view const& col,
                                   null_equality nulls_equal,
                                   nan_equality nans_equal,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto flatten_col    = col.get_sliced_child(stream);
  auto distinct_table = detail::distinct(table_view{{flatten_col}},
                                         std::vector<size_type>{0},
                                         duplicate_keep_option::KEEP_ANY,
                                         nulls_equal,
                                         nans_equal,
                                         stream,
                                         mr);
  return std::make_unique<list_scalar>(std::move(distinct_table->get_column(0)), true, stream, mr);
}

}  // namespace reduction
}  // namespace cudf
