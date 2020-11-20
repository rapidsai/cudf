/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace {
// Returns true if the mask is true for index i in at least keep_threshold
// columns
struct valid_table_filter {
  __device__ inline bool operator()(cudf::size_type i)
  {
    auto valid = [i](auto column_device_view) { return column_device_view.is_valid(i); };

    auto count =
      thrust::count_if(thrust::seq, keys_device_view.begin(), keys_device_view.end(), valid);

    return (count >= keep_threshold);
  }

  valid_table_filter()  = delete;
  ~valid_table_filter() = default;

  valid_table_filter(cudf::table_device_view const& keys_device_view,
                     cudf::size_type keep_threshold)
    : keep_threshold(keep_threshold), keys_device_view(keys_device_view)
  {
  }

 protected:
  cudf::size_type keep_threshold;
  cudf::size_type num_columns;
  cudf::table_device_view keys_device_view;
};

}  // namespace

namespace cudf {
namespace detail {
/*
 * Filters a table to remove null elements.
 */
std::unique_ptr<table> drop_nulls(table_view const& input,
                                  std::vector<size_type> const& keys,
                                  cudf::size_type keep_threshold,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto keys_view = input.select(keys);
  if (keys_view.num_columns() == 0 || keys_view.num_rows() == 0 || not cudf::has_nulls(keys_view)) {
    return std::make_unique<table>(input, stream, mr);
  }

  auto keys_device_view = cudf::table_device_view::create(keys_view, stream);

  return cudf::detail::copy_if(
    input, valid_table_filter{*keys_device_view, keep_threshold}, stream, mr);
}

}  // namespace detail

/*
 * Filters a table to remove null elements.
 */
std::unique_ptr<table> drop_nulls(table_view const& input,
                                  std::vector<size_type> const& keys,
                                  cudf::size_type keep_threshold,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::drop_nulls(input, keys, keep_threshold, rmm::cuda_stream_default, mr);
}
/*
 * Filters a table to remove null elements.
 */
std::unique_ptr<table> drop_nulls(table_view const& input,
                                  std::vector<size_type> const& keys,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::drop_nulls(input, keys, keys.size(), rmm::cuda_stream_default, mr);
}

}  // namespace cudf
