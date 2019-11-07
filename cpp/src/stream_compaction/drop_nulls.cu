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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/copy_if.cuh>

namespace {

// Returns true if the mask is true for index i in at least keep_threshold
// columns
struct valid_table_filter
{
  __device__ inline
  bool operator()(cudf::size_type i)
  {
    auto valid = [i](auto column_device_view) {
      return column_device_view.is_valid(i);
    };

    auto count =
      thrust::count_if(thrust::seq, keys_device_view.begin(), keys_device_view.end(), valid);

    return (count >= keep_threshold);
  }

  static auto create(cudf::table_device_view const& keys,
                     cudf::size_type keep_threshold,
                     cudaStream_t stream = 0)
  {
    //auto keys_device_view = cudf::table_device_view::create(keys, stream);

    auto deleter = [stream](valid_table_filter* f) { f->destroy(); };
    std::unique_ptr<valid_table_filter, decltype(deleter)> p {
      new valid_table_filter(keys, keys.num_columns(), keep_threshold),
      deleter
    };
    
    return p;
  }

  __host__ void destroy() {
    delete this;
  }

  valid_table_filter() = delete;
  ~valid_table_filter() = default;

protected:


  valid_table_filter(cudf::table_device_view const& keys_device_view,
                     cudf::size_type num_columns,
                     cudf::size_type keep_threshold)
  : keep_threshold(keep_threshold),
    num_columns(num_columns),
    keys_device_view(keys_device_view) {}

  cudf::size_type keep_threshold;
  cudf::size_type num_columns;
  cudf::table_device_view keys_device_view;
};

}  // namespace

namespace cudf {
namespace experimental {
namespace detail {

/*
 * Filters a table to remove null elements.
 */
std::unique_ptr<experimental::table> drop_nulls(table_view const& input,
                 table_view const& keys,
                 cudf::size_type keep_threshold,
                 rmm::mr::device_memory_resource *mr,
                 cudaStream_t stream) {
  if (keys.num_columns() == 0 || keys.num_rows() == 0 ||
      not cudf::has_nulls(keys)) {
      return std::make_unique<table>(input, stream, mr);
  }

  CUDF_EXPECTS(keys.num_rows() <= input.num_rows(),
               "Column size mismatch");

  auto keys_device_view = cudf::table_device_view::create(keys, stream);
  auto filter = valid_table_filter::create(*keys_device_view, keep_threshold);

  return cudf::experimental::detail::copy_if(input, *filter.get(), mr, stream);
}

} //namespace detail

/*
 * Filters a table to remove null elements.
 */
std::unique_ptr<experimental::table> drop_nulls(table_view const& input,
                 table_view const& keys,
                 cudf::size_type keep_threshold,
                 rmm::mr::device_memory_resource *mr) {
    return cudf::experimental::detail::drop_nulls(input, keys, keep_threshold, mr);
}
/*
 * Filters a table to remove null elements.
 */
std::unique_ptr<experimental::table> drop_nulls(table_view const &input,
                 table_view const &keys,
                 rmm::mr::device_memory_resource *mr)
{
    return cudf::experimental::detail::drop_nulls(input, keys, keys.num_columns(), mr);
}

} //namespace experimental
} //namespace cudf
